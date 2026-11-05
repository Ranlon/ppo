"""Native per-token log-softmax/gather with bounded intermediate backward memory."""
import torch
from trl.core import logprobs_from_logits as native_logprobs
CHUNK=32

class SelectedLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,labels):
        assert logits.ndim==3 and labels.shape==logits.shape[:2]
        ctx.save_for_backward(logits,labels)
        output=torch.empty(labels.shape,device=logits.device,dtype=logits.dtype)
        for batch,row in enumerate(logits):
            for start in range(0,row.shape[0],CHUNK):
                end=start+CHUNK
                logp=torch.nn.functional.log_softmax(row[start:end],dim=-1)
                output[batch,start:end]=logp.gather(-1,labels[batch,start:end,None]).squeeze(-1)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        logits,labels=ctx.saved_tensors
        gradient=torch.empty_like(logits)
        for batch,row in enumerate(logits):
            for start in range(0,row.shape[0],CHUNK):
                end=start+CHUNK
                # Recompute the same native row-wise operation; no softmax approximation.
                logp=torch.nn.functional.log_softmax(row[start:end],dim=-1)
                grad_logp=torch.zeros_like(logp)
                grad_logp.scatter_(-1,labels[batch,start:end,None],grad_output[batch,start:end,None])
                gradient[batch,start:end]=torch.ops.aten._log_softmax_backward_data(
                    grad_logp,logp,-1,logits.dtype)
        return gradient,None

def selected_logprobs(logits,labels,gather=True):
    if not gather:return native_logprobs(logits,labels,gather=False)
    return SelectedLogProb.apply(logits,labels)

def install():
    import importlib
    core=importlib.import_module('trl.core');ppo=importlib.import_module('trl.trainer.ppo_trainer')
    assert core.logprobs_from_logits is native_logprobs
    core.logprobs_from_logits=selected_logprobs
    ppo.logprobs_from_logits=selected_logprobs
