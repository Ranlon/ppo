"""CPU-only comparison with the actual vendored LF preprocessing pipeline."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
os.environ['DISABLE_VERSION_CHECK']='1'
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'runtime'))
from project import read

def main():
    from transformers import AutoTokenizer,Seq2SeqTrainingArguments
    from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments,ModelArguments
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    root=Path(args.run_dir).resolve();request=read(root/'REQUESTED_CONFIG.json')
    model_args=ModelArguments(model_name_or_path=request['requested_lf_arguments']['model_name_or_path'])
    tokenizer=AutoTokenizer.from_pretrained(model_args.model_name_or_path,local_files_only=True)
    data_args=DataArguments(dataset='gsm8k_ppo',dataset_dir=str(root/'data'),template='qwen3_nothink',
        cutoff_len=1024,preprocessing_num_workers=1,overwrite_cache=True)
    training_args=Seq2SeqTrainingArguments(output_dir=str(root/'preprocessing_verify'),use_cpu=True,
        report_to=[],do_train=True,remove_unused_columns=False)
    template=get_template_and_fix_tokenizer(tokenizer,data_args)
    dataset=get_dataset(template,model_args,data_args,training_args,'ppo',tokenizer)['train_dataset']
    rows=[json.loads(line) for line in (root/'PPO_PROMPT_METADATA.jsonl').read_text().splitlines()]
    assert len(dataset)==len(rows)==7473
    for index,ids in enumerate(dataset['input_ids']):
        assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==rows[index]['prompt_token_sha256'],index
    print(json.dumps(dict(passed=True,prompt_hashes_equal=7473,maximum_prompt_tokens=max(rows[i]['prompt_tokens'] for i in range(len(rows))),gpu_used=False)))

if __name__=='__main__':main()
