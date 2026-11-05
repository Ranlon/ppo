"""Shared-base frozen SFT reference context, to be validated before PPO use."""
import contextlib
import random
import numpy as np
import torch
from peft.utils.save_and_load import get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file


def restore_exact_adapter(model,checkpoint,name):
    """Copy original FP32 state only after the destination adapter is FP32."""
    from pathlib import Path
    state=load_file(str(Path(checkpoint)/'adapter_model.safetensors'),device='cpu')
    for key,param in model.named_parameters():
        if 'lora_' in key and '.'+name+'.' in key:
            param.data=param.data.to(torch.float32)
    set_peft_model_state_dict(model,state,adapter_name=name)
    actual=get_peft_model_state_dict(model,adapter_name=name)
    assert set(actual)==set(state)
    assert all(torch.equal(actual[key].detach().cpu(),value) for key,value in state.items())


class FrozenSFTReference:
    def __init__(self,model,checkpoint,name='sft_reference'):
        self.model=model;self.name=name
        assert list(model.active_adapters)==['default']
        assert name not in model.peft_config
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        try:
            model.load_adapter(str(checkpoint),adapter_name=name,is_trainable=False)
            actor=get_peft_model_state_dict(model,adapter_name='default')
            loaded=get_peft_model_state_dict(model,adapter_name=name)
            self.initial_load_equal_to_actor=all(torch.equal(actor[key],loaded[key]) for key in actor)
            self.initial_load_equal_to_bf16_rounded_actor=all(torch.equal(actor[key].to(torch.bfloat16).to(torch.float32),loaded[key]) for key in actor)
            restore_exact_adapter(model,checkpoint,name)
            for key,param in model.named_parameters():
                if 'lora_' in key and '.'+name+'.' in key:param.requires_grad_(False)
            assert list(model.active_adapters)==['default']
        finally:
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
            if rng[3]:torch.cuda.set_rng_state_all(rng[3])

    @contextlib.contextmanager
    def context(self):
        assert not getattr(self.model,'_ppo4_merge_active',False),'Reference cannot run while actor weights are merged'
        assert list(self.model.active_adapters)==['default']
        saved={key:param.requires_grad for key,param in self.model.named_parameters() if 'lora_' in key}
        try:
            self.model.set_adapter(self.name)
            # PEFT set_adapter enables selected parameters; the reference must stay frozen.
            for key,param in self.model.named_parameters():
                if 'lora_' in key:param.requires_grad_(False)
            yield
        finally:
            self.model.set_adapter('default')
            for key,param in self.model.named_parameters():
                if key in saved:param.requires_grad_(saved[key])
