"""Fixed TRAINING questions for EOS/quality gates; never restores full SFT test eval."""
import os
os.environ.update(DISABLE_VERSION_CHECK='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import argparse,hashlib,json,random,sys,time
from project import run_root,VENDOR
ROOT=run_root()
sys.path.insert(0,str(VENDOR))

def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from adapter_io import install
    install()
    from peft import PeftModel
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    from frozen_sft_reference import restore_exact_adapter
    from numeric_reward import numeric_answer
    parser=argparse.ArgumentParser();parser.add_argument('--adapter',required=True);parser.add_argument('--name',required=True)
    args=parser.parse_args();assert args.name.replace('_','').isalnum()
    out=ROOT/'quality'/args.name;out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in (ROOT/'data/train.jsonl').read_text().splitlines()]
    indices=random.Random(20261030).sample(range(7473),32)
    base=json.loads((ROOT/'REQUESTED_CONFIG.json').read_text())['requested_lf_arguments']['model_name_or_path']
    set_seed(42);torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    template=get_template_and_fix_tokenizer(tokenizer,DataArguments(template='qwen3_nothink',cutoff_len=1024))
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(model,args.adapter,is_trainable=True).to('cuda').eval()
    restore_exact_adapter(model,args.adapter,'default');model.requires_grad_(False)
    stopped=0;correct=0;empty=0;records=[];started=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),(out/'rows.jsonl').open('x') as log:
        for index in indices:
            row=rows[index];ids,_=template.encode_oneturn(tokenizer,[dict(role='user',content=row['question']),dict(role='assistant',content='')],'','')
            x=torch.tensor([ids],device='cuda')
            y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True)
            generated=y[0,len(ids):].tolist();text=tokenizer.decode(generated,skip_special_tokens=True)
            eos=generated[-1]==tokenizer.eos_token_id;hit=numeric_answer(text)==numeric_answer(row['answer'],gold=True)
            stopped+=eos;correct+=hit;empty+=not text.strip()
            record=dict(training_index=index,response=text,eos=eos,correct=hit,tokens=len(generated))
            records.append(record);log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
            (out/'PROGRESS.json').write_text(json.dumps(dict(rows=len(records),total=32,eos=stopped,correct=correct))+'\n')
            if len(records)-stopped>=4:break # Can no longer reach 90% EOS on 32 fixed questions.
    passed=len(records)==32 and stopped>=29 and correct>=16 and empty==0
    result=dict(passed=passed,training_questions_only=True,independent_test_used=False,samples=len(records),planned=32,
        eos_stopped=stopped,correct=correct,empty=empty,seconds=time.monotonic()-started,indices=indices,
        adapter_sha256=hashlib.sha256((Path(args.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest(),
        gate='At least 29/32 natural EOS, at least 16/32 numeric correct, no empty response',
        mean_response_tokens=sum(r['tokens'] for r in records)/len(records))
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not passed:raise RuntimeError('Training-question quality gate failed')

if __name__=='__main__':main()
