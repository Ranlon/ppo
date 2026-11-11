#!/usr/bin/env python3
"""Fixed training-question probe, evaluated fully before and after short PPO."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(DISABLE_VERSION_CHECK='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(args.source / 'runtime'))
    import project
    from project import read, write, sha
    from adapter_io import install
    install()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import PeftModel
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    from frozen_sft_reference import restore_exact_adapter
    from numeric_reward import numeric_answer

    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    plan = read(args.plan)
    request = read(args.run_dir / 'REQUESTED_CONFIG.json')
    data = args.run_dir / 'data/train.jsonl'
    assert sha(data) == plan['train_sha256']
    with data.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    set_seed(42)
    torch.set_num_threads(2)
    base = request['requested_lf_arguments']['model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template='qwen3_nothink', cutoff_len=1024))
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                        attn_implementation='sdpa', local_files_only=True)
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=True).to('cuda').eval()
    restore_exact_adapter(model, args.adapter, 'default')
    model.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), \
            (args.output / 'rows.jsonl').open('x', encoding='utf-8') as log:
        for index in plan['indices']:
            row = rows[index]
            ids, _ = template.encode_oneturn(tokenizer, [dict(role='user', content=row['question']),
                                                        dict(role='assistant', content='')], '', '')
            x = torch.tensor([ids], device='cuda')
            y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                max_new_tokens=512, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, use_cache=True)
            generated = y[0, len(ids):].tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            predicted = numeric_answer(text)
            gold = numeric_answer(row['answer'], gold=True)
            record = dict(training_index=index, correct=predicted is not None and predicted == gold,
                eos=bool(generated and generated[-1] == tokenizer.eos_token_id),
                empty=not text.strip(), tokens=len(generated), prediction=predicted, gold=gold, response=text)
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=True) + '\n')
            log.flush()
            write(args.output / 'PROGRESS.json', dict(samples=len(records), total=len(plan['indices']),
                                                     correct=sum(r['correct'] for r in records)))
    count = len(records)
    result = dict(samples=count, correct=sum(r['correct'] for r in records),
        numeric_accuracy=sum(r['correct'] for r in records)/count,
        natural_eos_rate=sum(r['eos'] for r in records)/count,
        empty=sum(r['empty'] for r in records), mean_response_tokens=sum(r['tokens'] for r in records)/count,
        seconds=time.perf_counter()-started, adapter=str(args.adapter.resolve()),
        adapter_sha256=sha(args.adapter / 'adapter_model.safetensors'), indices=plan['indices'],
        training_questions_only=True, excluded_short_ppo_training_prefix=True,
        evaluation='native SDPA, greedy, batch1, max512; no early probe termination',
        limitation='Small training-set diagnostic, not held-out accuracy or proof of no quality loss')
    write(args.output / 'RESULT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
