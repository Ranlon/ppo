"""Fixed native greedy evaluation of final SFT and final PPO on untouched GSM8K test."""
import os
os.environ.update(DISABLE_VERSION_CHECK='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from project import run_root,VENDOR
ROOT=run_root()
sys.path.insert(0,str(VENDOR))


def numeric(text, gold=False):
    text = text.split('</think>')[-1]
    text = re.sub(r'<\|[^>]+\|>', '', text).strip()
    if '####' in text:
        match = re.match(r'\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)', text.rsplit('####', 1)[1])
    elif gold:
        return None
    else:
        boxed = re.findall(r'\\boxed\{\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)\s*\}', text)
        values = boxed or re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
        match = re.fullmatch(r'([-+]?\d[\d,]*(?:\.\d+)?)', values[-1] if values else '')
    if not match: return None
    try:
        value = Decimal(match.group(1).replace(',', ''))
        return str(value.normalize()) if value.is_finite() else None
    except InvalidOperation: return None


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    parser = argparse.ArgumentParser(); parser.add_argument('--label', choices=['SFT', 'PPO'], required=True)
    parser.add_argument('--model', required=True); args = parser.parse_args()
    output = ROOT / 'evaluation' / args.label; output.mkdir(parents=True, exist_ok=False)
    request = json.loads((ROOT / 'REQUESTED_CONFIG.json').read_text())
    manifest = json.loads((ROOT / 'data/UPSTREAM_MANIFEST.json').read_text())
    test = ROOT / 'data/test.jsonl'
    assert hashlib.sha256(test.read_bytes()).hexdigest() == manifest['test']['sha256']
    rows = [json.loads(line) for line in test.read_text().splitlines()]; assert len(rows) == 1319
    set_seed(42); torch.set_num_threads(4)
    base = args.model
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template='qwen3_nothink', cutoff_len=1024))
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    model = model.to('cuda').eval()
    model.requires_grad_(False)
    eos = list(dict.fromkeys([tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids))
    started = time.monotonic(); correct = 0; strict = 0; capped = 0; tokens = 0
    from full_support import full_model_identity
    identity = dict(label=args.label, model=args.model,
        model_sha256=full_model_identity(args.model)['sha256'],
        test_sha256=manifest['test']['sha256'], examples=1319, do_sample=False, max_new_tokens=512,
        batch_size=1, template='qwen3_nothink', prompt='Original question only',
        native_SDPA=True, full_model_BF16_evaluation=True, no_test_driven_checkpoint_selection=True,
        eos_token_ids=eos, pad_token_id=tokenizer.pad_token_id)
    (output / 'IDENTITY.json').write_text(json.dumps(identity, indent=2)+'\n')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), (output / 'rows.jsonl').open('x') as handle:
        for index, row in enumerate(rows):
            ids, _ = template.encode_oneturn(tokenizer, [dict(role='user', content=row['question']), dict(role='assistant', content='')], '', '')
            assert len(ids) <= 1024
            x = torch.tensor([ids], device='cuda')
            y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=512,
                do_sample=False, num_beams=1, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.0, use_cache=True)
            generated = y[0, len(ids):].tolist(); text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            gold = numeric(row['answer'], gold=True); predicted = numeric(text)
            assert gold is not None
            hit = predicted is not None and predicted == gold
            strict_hit = hit and '####' in text
            cap = len(generated) == 512 and generated[-1] not in eos
            correct += hit; strict += strict_hit; capped += cap; tokens += len(generated)
            record = dict(index=index, question=row['question'], gold=gold, prediction=predicted,
                response=text, numeric_correct=hit, strict_correct=strict_hit, response_tokens=len(generated), token_cap=cap, natural_eos=generated[-1] in eos,
                stop_token_id=generated[-1], format_present='####' in text, empty=not text.strip())
            handle.write(json.dumps(record, ensure_ascii=False)+'\n'); handle.flush()
            if (index+1)%10 == 0 or index+1 == len(rows):
                temp = output/'PROGRESS.json.tmp'; temp.write_text(json.dumps(dict(rows=index+1, total=1319,
                    elapsed_seconds=time.monotonic()-started, numeric_correct=correct, strict_correct=strict))+'\n')
                os.replace(temp, output/'PROGRESS.json')
    result = dict(passed=True, **identity, numeric_correct=correct, strict_correct=strict,
        numeric_accuracy=correct/1319, strict_accuracy=strict/1319, token_cap_outputs=capped,
        generated_tokens=tokens, seconds=time.monotonic()-started,
        rows_sha256=hashlib.sha256((output/'rows.jsonl').read_bytes()).hexdigest())
    (output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
