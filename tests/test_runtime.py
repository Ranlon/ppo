"""CPU checks using synthetic tensors and a randomly initialized tiny Qwen3.

No pretrained weights, datasets, or GPU jobs are used.
"""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import dataclasses
import json
import os
import random
import sys
import tempfile
import unittest
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'runtime'))
from project import read,write,sha
from cli import profile,read_rows
from head_runtime import BF16LogProb,make_masks
from portable_rollout import ExactMerge
from checkpoint_io import save_complete,load_complete,rng_state,restore_rng
from compare_resume import compare


def tiny():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from peft import LoraConfig,get_peft_model
    config=Qwen3Config(vocab_size=53,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True,
        attention_dropout=0.,pad_token_id=0,eos_token_id=2)
    model=Qwen3ForCausalLM(config).to(torch.bfloat16)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head']))
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'lora_' in name:
                param.data=param.data.float()
                param.normal_(0,.02)
    return model.eval()


class RuntimeTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);random.seed(42);np.random.seed(42)

    def test_configuration(self):
        self.assertEqual({p.stem for p in (PROJECT/'configs').glob('*.yaml') if p.stem!='common'}, {'baseline','optimized'})
        for name in ('baseline','optimized'):
            config=profile(name);common=config['common']
            self.assertEqual(config,read(PROJECT/f'configs/{name}.resolved.json'))
            self.assertEqual(common['sft_updates'],935)
            self.assertEqual(common['ppo_optimizer_updates'],9964)
            self.assertEqual(common['ppo']['ppo_epochs'],4)
            self.assertEqual(common['ppo']['max_new_tokens'],512)
        self.assertFalse(profile('baseline')['runtime']['python_gc_per_batch'])
        self.assertEqual(profile('optimized')['runtime']['rollout'],'merged_compiled')

    def test_synthetic_data_contract(self):
        from numeric_reward import numeric_answer
        self.assertEqual(numeric_answer('explanation 7 then \\boxed{1,234}'),'1234')
        self.assertEqual(numeric_answer('last 9\n#### 42'),'42')
        self.assertIsNone(numeric_answer('42',gold=True))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'toy.jsonl'
            path.write_text(json.dumps(dict(question='toy',answer='#### 3'))+'\n')
            self.assertEqual(len(read_rows(path,1)),1)
            with self.assertRaises(AssertionError):read_rows(path,2)

    def test_staged_probabilities_and_backward(self):
        for dtype in (torch.bfloat16,torch.float64):
            logits=torch.randn(3,79,53,dtype=dtype,requires_grad=True)
            original=logits.detach().clone().requires_grad_(True)
            labels=torch.randint(0,53,(3,79))
            mask=torch.zeros((3,79),dtype=torch.long);mask[:,34:57]=1
            selected,entropy=BF16LogProb.apply(logits,labels,mask,True)
            dense=original.double() if dtype==torch.float64 else original.float()
            reference=dense.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
            self.assertTrue(torch.equal(selected[mask.bool()],reference[mask.bool()]))
            weights=torch.randn(3,79)*mask
            (selected*weights).sum().backward();(reference*weights).sum().backward()
            self.assertTrue(torch.equal(logits.grad,original.grad))
            self.assertTrue(torch.equal(selected[:,:32],torch.zeros_like(selected[:,:32])))
            self.assertTrue(torch.isfinite(entropy).all())

    def test_masks_left_and_right_padding(self):
        queries=[torch.ones(3),torch.ones(2),torch.ones(4)]
        responses=[torch.ones(2),torch.ones(3),torch.ones(1)]
        attention=torch.tensor([[0,1,1,1,1,1],[1,1,1,1,1,0],[1,1,1,1,1,0]])
        masks=make_masks(queries,responses,attention)
        self.assertEqual(masks.sum(-1).tolist(),[2,3,1])
        self.assertEqual(masks[0].nonzero().flatten().tolist(),[3,4])

    def test_full_head_value_branch_and_all_lora_gradients(self):
        from trl.models.modeling_value_head import ValueHead
        from head_runtime import batched_forward
        model=tiny();model.config.summary_dropout_prob=0.
        value_head=ValueHead(model.config).float()
        raw=SimpleNamespace(pretrained_model=model,v_head=value_head)
        inputs=dict(input_ids=torch.randint(3,53,(3,12)),attention_mask=torch.ones(3,12,dtype=torch.long))
        queries=[inputs['input_ids'][i,:4] for i in range(3)]
        responses=[inputs['input_ids'][i,4:] for i in range(3)]
        trainer=SimpleNamespace(finetuning_args=SimpleNamespace(ppo_batch_python_gc=True),
            _m9_head_config=dict(mode='bf16_masked',chunk=32,checkpoint='all',trunk='padded'),
            accelerator=SimpleNamespace(unwrap_model=lambda x:x,num_processes=1),config=SimpleNamespace(mini_batch_size=3),
            amp_context=contextlib.nullcontext())
        weights=torch.randn(3,11);value_weights=torch.randn(3,11)
        output=model(**inputs,output_hidden_states=True,use_cache=False,return_dict=True)
        baseline=output.logits[:,:-1].float().log_softmax(-1).gather(-1,inputs['input_ids'][:,1:,None]).squeeze(-1)
        values=value_head(output.hidden_states[-1]).squeeze(-1)[:,:-1]
        masks=make_masks(queries,responses,inputs['attention_mask'])[:,:-1]
        ((baseline*weights*masks).sum()+(values*value_weights*masks).sum()).backward()
        gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        value_gradients={n:p.grad.clone() for n,p in value_head.named_parameters()}
        model.zero_grad();value_head.zero_grad()
        selected,_,actual_values,actual_masks=batched_forward(trainer,raw,queries,responses,inputs,return_logits=True)
        self.assertTrue(torch.equal(selected*masks,baseline.detach()*masks))
        self.assertTrue(torch.equal(actual_values,values.detach()))
        ((selected*weights*actual_masks).sum()+(actual_values*value_weights*actual_masks).sum()).backward()
        for name,p in model.named_parameters():
            if name in gradients:self.assertTrue(torch.equal(p.grad,gradients[name]),name)
        for name,p in value_head.named_parameters():self.assertTrue(torch.equal(p.grad,value_gradients[name]),name)

    def test_merge_restore_tied_head_latest_adapter_and_exceptions(self):
        from adapter_io import install
        install()
        from frozen_sft_reference import FrozenSFTReference
        model=tiny()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reference=FrozenSFTReference(model,tmp)
            initial={name:p.detach().clone() for name,p in model.named_parameters()}
            identities={name:id(p) for name,p in model.named_parameters()}
            embedding=model.get_input_embeddings().weight
            for backup in ('gpu','cpu'):
                merge=ExactMerge(model,backup,chunk_rows=7)
                for failure in (False,True):
                    try:
                        with merge.active():
                            self.assertIs(model.get_input_embeddings().weight,embedding)
                            self.assertTrue(torch.equal(embedding,initial['base_model.model.model.embed_tokens.weight']))
                            self.assertIsNot(model.get_output_embeddings().base_layer.weight,embedding)
                            with self.assertRaises(AssertionError):
                                with reference.context():pass
                            for _,module,weight,_,is_head,_,_ in merge.entries:
                                dense=weight if not is_head else embedding
                                a=module.lora_A['default'].weight;b=module.lora_B['default'].weight
                                expected=(initial[next(n for n,p in model.named_parameters() if p is weight)].float() +
                                    b.float()@a.float()*module.scaling['default']).to(dense.dtype) if not is_head else (
                                    embedding.float()+b.float()@a.float()*module.scaling['default']).to(dense.dtype)
                                self.assertTrue(torch.equal(module.base_layer.weight,expected))
                            if failure:raise RuntimeError('synthetic interruption')
                    except RuntimeError as error:
                        if str(error)!='synthetic interruption':raise
                    for name,p in model.named_parameters():
                        self.assertTrue(torch.equal(p,initial[name]),name)
                        self.assertEqual(id(p),identities[name])
                with reference.context():
                    self.assertEqual(list(model.active_adapters),['sft_reference'])
            module=model.get_output_embeddings()
            merge=ExactMerge(model,'cpu',7)
            with merge.active():before=module.base_layer.weight.detach().clone()
            with torch.no_grad():module.lora_B['default'].weight.add_(.5)
            with merge.active():after=module.base_layer.weight.detach().clone()
            self.assertFalse(torch.equal(before,after),'Merged generation must read the latest LoRA')
            merge.verify_exact()

    def test_full_checkpoint_roundtrip_rng_and_next_update(self):
        @dataclasses.dataclass
        class State:
            global_step:int=1
            log_history:list=dataclasses.field(default_factory=list)
        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.randn(4,4))
                self.pretrained_model=SimpleNamespace(active_adapters=['default'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'runs/control';out.mkdir(parents=True)
            actor=Actor();optimizer=torch.optim.AdamW(actor.parameters(),lr=1e-3)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:1-i/20)
            actor.weight.square().sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad()
            trainer=SimpleNamespace(root=root,output=out,state=State(),state_save_every=250,model=actor,
                accelerator=SimpleNamespace(unwrap_model=lambda x:x,scaler=None),optimizer=optimizer,
                lr_scheduler=scheduler,kl_ctl=SimpleNamespace(value=.2),config_sha='toy',sft_sha='toy-sft',
                reward_revision='toy-reward',variant='native',strict=False,rows=[dict(batch=1)],
                rollouts=[dict(index=i) for i in range(3)],initial_trainable_sha={'weight':'initial'},
                elapsed_before_resume=0.,program_start=0.,sample_order=[2,0,1],zero_reward_streak=0,
                quality_stop_reason=None,check_frozen_reference=lambda:None)
            save_complete(trainer,force=True)
            pointer=read(out/'LATEST_CHECKPOINT.json');state=torch.load(pointer['path'],weights_only=False)
            expected=(random.random(),np.random.rand(),torch.rand(3))
            x=torch.randn(4,4);(actor.weight*x).square().sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad()
            expected_weight=actor.weight.detach().clone();expected_optimizer=optimizer.state_dict()
            resumed=root/'runs/resumed';resumed.mkdir();trainer.output=resumed
            load_complete(trainer,pointer['path']);restore_rng(trainer._resume_rng)
            actual=(random.random(),np.random.rand(),torch.rand(3));compare(expected,actual,'rng')
            x2=torch.randn(4,4);self.assertTrue(torch.equal(x,x2))
            (actor.weight*x2).square().sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad()
            self.assertTrue(torch.equal(expected_weight,actor.weight))
            compare(expected_optimizer,optimizer.state_dict(),'next_adam_update')
            self.assertEqual(trainer.kl_ctl.value,.2)


if __name__=='__main__':unittest.main(verbosity=2)
