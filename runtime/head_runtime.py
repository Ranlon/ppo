"""Native full-shape BF16 head with response-block FP32 probability staging.

Only single-device vanilla Linear LoRA is supported. The PPO objective and
full-vocabulary normalization are unchanged. Outside-mask placeholders are
explicitly zero and are never represented as measured prompt probabilities.
"""
from dataclasses import dataclass
import contextlib
import math
import torch
from trl.core import entropy_from_logits as native_entropy


@dataclass
class HeadStatistics:
    entropy: torch.Tensor
    computed_positions: int
    total_positions: int
    mode: str


def amp_for(tensor):
    if tensor.device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def upcast(logits):
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


@torch.no_grad()
def entropy_rows(logits):
    # Keep the existing native32 entropy formula and full vocabulary reduction.
    return torch.cat([native_entropy(logits[i:i+32]) for i in range(0, len(logits), 32)])


class BF16LogProb(torch.autograd.Function):
    """Preserve the original full-shape head GEMM and its native gradient.

    Only FP32 cast, log-softmax and their backward buffers are staged. The
    full-shaped logits and its incoming gradient remain in the head's BF16.
    """
    @staticmethod
    def forward(ctx, logits, labels, mask, response_only):
        assert logits.ndim==3 and logits.shape[:2]==labels.shape
        ctx.save_for_backward(logits, labels)
        ctx.blocks=[]
        mask_rows=mask.detach().cpu().tolist() if response_only else None
        dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32
        output=torch.zeros(labels.shape,device=logits.device,dtype=dtype)
        entropy=torch.zeros_like(output)
        for row in range(logits.shape[0]):
            for start in range(0, logits.shape[1], 32):
                end=min(start+32,logits.shape[1])
                if response_only and not any(mask_rows[row][start:end]):continue
                ctx.blocks.append((row,start,end))
                x=upcast(logits[row,start:end])
                logp=torch.nn.functional.log_softmax(x,dim=-1)
                output[row,start:end]=logp.gather(-1,labels[row,start:end,None]).squeeze(-1)
                entropy[row,start:end]=native_entropy(x)
        ctx.mark_non_differentiable(entropy)
        return output,entropy

    @staticmethod
    def backward(ctx, grad_output, grad_entropy):
        logits,labels=ctx.saved_tensors
        # Native FloatBackward would cast this gradient to BF16 before the
        # original head backward. Do that cast per block, without a global
        # FP32 gradient/logits allocation or any head-GEMM recomputation.
        gradient=torch.zeros_like(logits)
        for row,start,end in ctx.blocks:
            x=upcast(logits[row,start:end])
            logp=torch.nn.functional.log_softmax(x,dim=-1)
            grad_logp=torch.zeros_like(logp)
            grad_logp.scatter_(-1,labels[row,start:end,None],grad_output[row,start:end,None])
            native_gradient=torch.ops.aten._log_softmax_backward_data(grad_logp,logp,-1,x.dtype)
            gradient[row,start:end]=native_gradient.to(logits.dtype)
        return gradient,None,None,None


def head_probabilities(hidden,head,labels,mask,mode,chunk):
    assert mode=='bf16_masked' and chunk==32
    with amp_for(hidden):logits=head(hidden)
    selected,entropy=BF16LogProb.apply(logits,labels,mask,True)
    return selected,entropy,labels.numel()

def apply_checkpoint_policy(raw,policy):
    assert policy=='all'
    layers=raw.pretrained_model.get_base_model().model.layers
    for layer in layers:layer.gradient_checkpointing=True
    return len(layers)

def install_entropy_dispatch():
    import trl.trainer.ppo_trainer as ppo
    if getattr(ppo.entropy_from_logits, '_m9_dispatch', False):
        return
    previous = ppo.entropy_from_logits
    def entropy(logits):
        return logits.entropy if isinstance(logits, HeadStatistics) else previous(logits)
    entropy._m9_dispatch = True
    ppo.entropy_from_logits = entropy


def make_masks(queries, responses, attention_mask, response_masks=None):
    masks = torch.zeros_like(attention_mask)
    masks[:, :-1] = attention_mask[:, 1:]
    for j, (query, response) in enumerate(zip(queries, responses)):
        start = len(query) - 1
        if attention_mask[j, 0] == 0:
            start += int(attention_mask[j].nonzero()[0].item())
        end = start + len(response)
        masks[j, :start] = 0; masks[j, end:] = 0
        if response_masks is not None:
            rmask = torch.cat((torch.zeros_like(query), response_masks[j]))[1:]
            masks[j, start:end] *= rmask[start:end]
    return masks


def trunk_forward(raw,inputs,policy):
    assert policy=='padded'
    return raw.pretrained_model.get_base_model().model(**inputs,output_hidden_states=True,return_dict=True,use_cache=False).hidden_states[-1]

def batched_forward(trainer, model, queries, responses, model_inputs,
                    return_logits=False, response_masks=None):
    from llamafactory.extras.misc import torch_gc
    torch_gc(collect_python=not trainer.finetuning_args.ppo_batch_python_gc)
    config = trainer._m9_head_config
    raw = trainer.accelerator.unwrap_model(model)
    assert trainer.accelerator.num_processes == 1
    assert len(queries) == 3 and trainer.config.mini_batch_size == 3
    masks = make_masks(queries, responses, model_inputs['attention_mask'], response_masks)
    ncheckpoint = apply_checkpoint_policy(raw, config['checkpoint'])
    with trainer.amp_context:
        hidden = trunk_forward(raw, model_inputs, config['trunk'])
        ids = model_inputs['input_ids']
        labels = torch.cat((ids[:, 1:], torch.zeros_like(ids[:, :1])), dim=1)
        # Preserve all head positions, including the final unused position.
        # Match Qwen3ForCausalLM's full-range head view. It groups the
        # base/head-LoRA gradients before merging the value branch in BF16.
        head_hidden = hidden[:, slice(0, None), :]
        selected, entropy, npositions = head_probabilities(head_hidden,
            raw.pretrained_model.get_output_embeddings(), labels, masks,
            config['mode'], config['chunk'])
        # Same full shape, dtype handling, and RNG ordering as TRL's value head.
        if hidden.device != raw.v_head.summary.weight.device:
            hidden = hidden.to(raw.v_head.summary.weight.device)
        values = raw.v_head(hidden).squeeze(-1)
    statistics = HeadStatistics(entropy[:, :-1], npositions, ids.numel(), config['mode'])
    trainer._m9_last_audit = dict(mode=config['mode'], chunk=config['chunk'],
        checkpoint=config['checkpoint'], checkpoint_layers=ncheckpoint, trunk=config['trunk'],
        computed_head_positions=npositions, original_head_positions=ids.numel(),
        outside_response_placeholders=True,
        full_shape_bf16_logits=True,
        fp32_staging_only=True,
        full_vocabulary=True, head_lora_included=True, no_base_merge=True,
        selective_checkpoint_counters=getattr(raw,'_m9_checkpoint_counters',{}))
    return selected[:, :-1], statistics if return_logits else None, values[:, :-1], masks[:, :-1]
