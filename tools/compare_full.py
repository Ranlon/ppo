#!/usr/bin/env python3
"""Compare completed SFT coverage and a common PPO prefix, including cold compile."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(root, count):
    sft = read(root / 'sft/SUCCESS.json')
    with (root / 'runs/full/batches.jsonl').open() as handle:
        rows = [json.loads(line) for line in handle][:count]
    assert len(rows) == count and [r['batch'] for r in rows] == list(range(1, count + 1))
    assert all(r['update_calls'] == {'backward': 4, 'optimizer': 4} for r in rows)
    seconds = sum(r['wall_seconds'] for r in rows)
    tokens = sum(r['input_tokens'] + r['response_tokens'] for r in rows)
    result = dict(sft=sft, ppo=dict(batches=count, optimizer_updates=4 * count,
        seconds=seconds, tokens=tokens, tokens_per_second=tokens / seconds,
        response_tokens_per_second=sum(r['response_tokens'] for r in rows) / seconds,
        generation_seconds=sum(r['generation_seconds'] for r in rows),
        ppo_step_seconds=sum(r['ppo_step_seconds'] for r in rows),
        max_allocated_gib=max(r['peak_allocated_bytes'] for r in rows) / 2**30,
        full_dataset_completed=(read(root / 'runs/full/SUCCESS.json').get('full_dataset_completed', False)
            if (root / 'runs/full/SUCCESS.json').exists() else False)),
        total=dict(seconds=sft['train_runtime'] + seconds,
                   tokens_per_second=(sft['input_tokens'] + tokens) / (sft['train_runtime'] + seconds)))
    result['sft']['input_tokens_per_second'] = sft['input_tokens'] / sft['train_runtime']
    result['ppo']['quality_stopped'] = (read(root / 'runs/full/QUALITY_STOPPED.json')
        if (root / 'runs/full/QUALITY_STOPPED.json').exists() else None)
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--optimized', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=800)
    args = parser.parse_args()
    counts = []
    for root in [args.baseline, args.optimized]:
        with (root / 'runs/full/batches.jsonl').open() as handle:
            counts.append(sum(1 for _ in handle))
    count = min(args.batches, *counts)
    assert count > 0
    base, b_rows = summarize(args.baseline, count)
    opt, o_rows = summarize(args.optimized, count)
    assert all(b['indices'] == o['indices'] and b['query_sha256'] == o['query_sha256']
               for b, o in zip(b_rows, o_rows))
    assert base['sft']['full_dataset_completed'] and opt['sft']['full_dataset_completed']
    assert base['sft']['input_tokens'] == opt['sft']['input_tokens']
    result = dict(common_prefix_batches=count, baseline=base, optimized=opt,
                  timing='SFT train_runtime + sum PPO batch wall_seconds; includes cold compile, '
                         'excludes PPO startup and separate full-state checkpoint serialization',
                  experiment='Independent full SFT weights. ODB changes batches, update counts and schedule. '
                             'This is an end-to-end configuration comparison, not isolated kernel ablation.')
    lines = ['# Full Qwen3-0.6B comparison', '', f'PPO common prefix: {count} batches, including cold compile.', '',
             '| Metric | Baseline | Optimized | Optimized / baseline |', '|---|---:|---:|---:|']
    for label, b, o in [
        ('SFT train seconds', base['sft']['train_runtime'], opt['sft']['train_runtime']),
        ('SFT input token/s', base['sft']['input_tokens_per_second'], opt['sft']['input_tokens_per_second']),
        ('SFT optimizer updates', base['sft']['optimizer_updates'], opt['sft']['optimizer_updates']),
        ('PPO seconds', base['ppo']['seconds'], opt['ppo']['seconds']),
        ('PPO input + response token/s', base['ppo']['tokens_per_second'], opt['ppo']['tokens_per_second']),
        ('PPO response token/s', base['ppo']['response_tokens_per_second'], opt['ppo']['response_tokens_per_second']),
        ('SFT + PPO seconds', base['total']['seconds'], opt['total']['seconds']),
        ('Combined token/s', base['total']['tokens_per_second'], opt['total']['tokens_per_second'])]:
        lines.append(f'| {label} | {b:.3f} | {o:.3f} | {o / b:.3f}x |')
    lines.extend(['', result['timing'], '', result['experiment'], '',
                  'Quality stops: ' + json.dumps({k: result[k]['ppo']['quality_stopped']
                                                  for k in ['baseline', 'optimized']})])
    out = args.baseline.resolve().parent
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
