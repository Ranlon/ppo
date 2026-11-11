"""Export only trained LoRA tensors, including lm_head; frozen base stays external."""
def install():
    import peft
    import peft.peft_model as model_module
    import peft.utils.save_and_load as io
    native=io.get_peft_model_state_dict
    if getattr(native,'_repair_lora_only',False):return
    def lora_only(*args,**kwargs):
        kwargs['save_embedding_layers']=False
        return native(*args,**kwargs)
    lora_only._repair_lora_only=True
    io.get_peft_model_state_dict=lora_only
    model_module.get_peft_model_state_dict=lora_only
    peft.get_peft_model_state_dict=lora_only
