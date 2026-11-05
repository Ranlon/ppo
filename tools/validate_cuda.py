"""Target-GPU validation: full restore, reference isolation, EOS/logprob drift.

This deliberately allocates CPU snapshots for audit. Run before 200-batch PPO.
No training is performed. It records differences without a hard 0.1 threshold.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
os.environ.update(DISABLE_VERSION_CHECK='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'runtime'))
from project import read,write


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import PeftModel
    from adapter_io import install
    install()
    from frozen_sft_reference import FrozenSFTReference,restore_exact_adapter
    from portable_rollout import NativeRolloutAcceleration
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--sft',required=True)
    parser.add_argument('--tokens',type=int,choices=[16,64,512],default=64);parser.add_argument('--name',default='cuda_validation')
    args=parser.parse_args();root=Path(args.run_dir).resolve();os.environ['PPO4_RUN_ROOT']=str(root)
    assert args.name.replace('_','').isalnum()
    out=root/args.name;out.mkdir(exist_ok=False)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/'inductor');os.environ['TRITON_CACHE_DIR']=str(out/'triton')
    request=read(root/'REQUESTED_CONFIG.json');base=request['requested_lf_arguments']['model_name_or_path']
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    set_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    template=get_template_and_fix_tokenizer(tokenizer,DataArguments(template='qwen3_nothink',cutoff_len=1024))
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(model,args.sft,is_trainable=True).to('cuda').eval()
    restore_exact_adapter(model,args.sft,'default');reference=FrozenSFTReference(model,args.sft)
    rows=[json.loads(line) for line in (root/'data/train.jsonl').read_text().splitlines()]
    ids=[template.encode_oneturn(tokenizer,[dict(role='user',content=r['question']),dict(role='assistant',content='')],'','')[0] for r in rows]
    indices=sorted(range(len(ids)),key=lambda i:len(ids[i]),reverse=True)[:3]
    tokenizer.padding_side='left'
    inputs=tokenizer.pad([dict(input_ids=ids[i],attention_mask=[1]*len(ids[i])) for i in indices],return_tensors='pt').to('cuda')
    snapshots={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    identities={name:(id(p),p.data_ptr()) for name,p in model.named_parameters()}
    def exact_restore():
        for name,p in model.named_parameters():
            assert identities[name]==(id(p),p.data_ptr()),name
            assert torch.equal(p.detach().cpu(),snapshots[name]),name
        assert model.get_input_embeddings().weight is model.get_output_embeddings().base_layer.weight
        assert list(model.active_adapters)==['default']
    def logits():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model(**inputs,use_cache=False).logits[:,-1].float()
    with reference.context():ref_before=logits()
    native=logits()
    helper=NativeRolloutAcceleration(model,3,768)
    kwargs=dict(**inputs,max_new_tokens=args.tokens,min_new_tokens=args.tokens,do_sample=False,
                eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True)
    torch.cuda.reset_peak_memory_stats();records=[]
    for repeat in range(3):
        started=time.perf_counter();output=helper.generate(**kwargs);torch.cuda.synchronize();elapsed=time.perf_counter()-started
        exact_restore()
        records.append(dict(repeat=repeat,seconds=elapsed,tokens=3*args.tokens,
            output_sha256=hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest(),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        write(out/'PROGRESS.json',dict(repeats=records,runtime=helper.statistics()))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),helper.patch.active():altered=logits()
    exact_restore()
    with reference.context():ref_after=logits()
    assert torch.equal(ref_before,ref_after),'Frozen reference changed after rollout'
    # This probe forces the existing compiled decode to read a newer output head.
    head=model.get_output_embeddings();b=head.lora_B['default'].weight
    with torch.no_grad():b.add_(.25)
    try:
        changed=helper.generate(**kwargs)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),helper.patch.active():new_logits=logits()
        assert not torch.equal(altered,new_logits),'Latest LoRA was not read'
        # Compare the warmed compiled path against the same patched eager path.
        helper.cache.reset()
        with helper.suspend_hooks(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),helper.norm.active(),helper.patch.active():
            eager=model.generate(**kwargs,past_key_values=helper.cache,disable_compile=True)
        assert torch.equal(changed,eager),'Compiled decode differs from current patched eager model'
    finally:
        original_name=next(name for name,p in model.named_parameters() if p is b)
        with torch.no_grad():b.copy_(snapshots[original_name])
    exact_restore()
    logp=native.log_softmax(-1);altered_logp=altered.log_softmax(-1)
    result=dict(passed=True,gpu=torch.cuda.get_device_name(0),pressure_tokens=args.tokens,indices=indices,
        exact_restore=True,reference_unchanged=True,latest_lora_read=True,compiled_updated_decode_matches_patched_eager=True,
        eos_max_logprob_error=float((altered_logp[:,tokenizer.eos_token_id]-logp[:,tokenizer.eos_token_id]).abs().max()),
        all_vocab_max_logprob_error=float((altered_logp-logp).abs().max()),
        error_threshold_0_1_enforced=False,scope='rollout validation only; full PPO memory must also be tested',
        repeats=records,runtime=helper.statistics())
    write(out/'RESULT.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
