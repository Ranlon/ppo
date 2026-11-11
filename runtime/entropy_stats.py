"""Memory-bounded native entropy statistics; PPO's optimization loss is unchanged."""
import torch
from trl.core import entropy_from_logits as native_entropy

@torch.no_grad()
def chunked_entropy(logits,chunk_size=32):
    assert logits.ndim==3
    # Do not flatten a noncontiguous [B,T-1,V] view: that would copy the full logits.
    return torch.stack([torch.cat([native_entropy(row[start:start+chunk_size])
        for start in range(0,row.shape[0],chunk_size)]) for row in logits])

def install():
    import importlib
    module=importlib.import_module('trl.trainer.ppo_trainer')
    assert module.entropy_from_logits is native_entropy
    module.entropy_from_logits=chunked_entropy
