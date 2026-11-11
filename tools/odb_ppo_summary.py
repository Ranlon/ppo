#!/usr/bin/env python3
"""Summarize an ordinary-vs-ODB SFT short PPO experiment without declaring noninferiority."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def phase_stats(rows):
    if not rows:
        raise ValueError('No complete PPO batches')
    assert [r['batch'] for r in rows] == list(range(1, len(rows)+1))
    assert all(r['update_calls'] == {'backward': 4, 'optimizer': 4} for r in rows)
    seconds = sum(r['wall_seconds'] for r in rows)
    tokens = sum(r['input_tokens']+r['response_tokens'] for r in rows)
    responses = sum(r['response_tokens'] for r in rows)
    return dict(batches=len(rows), optimizer_updates=4*len(rows), batch_wall_seconds=seconds,
        seconds_per_batch=seconds/len(rows), input_plus_response_tokens_per_second=tokens/seconds,
        response_tokens_per_second=responses/seconds,
        response_tokens_per_sample=responses/(3*len(rows)),
        generation_seconds_per_batch=sum(r['generation_seconds'] for r in rows)/len(rows),
        ppo_step_seconds_per_batch=sum(r['ppo_step_seconds'] for r in rows)/len(rows),
        reward_rate=sum(r['reward_mean'] for r in rows)/len(rows),
        natural_eos_rate=sum(r['eos_stopped'] for r in rows)/(3*len(rows)),
        mean_k3=sum(r['k3_kl'] for r in rows)/len(rows), max_k3=max(r['k3_kl'] for r in rows),
        mean_loss=sum(r['loss'] for r in rows)/len(rows),
        max_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
        allocator_ooms=max(r['allocator_ooms'] for r in rows),
        allocator_retries=max(r['allocator_retries'] for r in rows),
        finite_logged_loss_and_kl=all(math.isfinite(r[k]) for r in rows for k in ['loss','k3_kl']))


def paired_accuracy(a, b):
    assert [r['training_index'] for r in a] == [r['training_index'] for r in b]
    return dict(samples=len(a), baseline_only_correct=sum(bool(x['correct']) and not y['correct'] for x,y in zip(a,b)),
        odb_only_correct=sum(bool(y['correct']) and not x['correct'] for x,y in zip(a,b)),
        both_correct=sum(bool(x['correct']) and bool(y['correct']) for x,y in zip(a,b)),
        neither_correct=sum(not x['correct'] and not y['correct'] for x,y in zip(a,b)))


def summarize(output):
    experiment = read(output/'EXPERIMENT.json')
    all_rows = {name: jsonl(output/name/'runs/short/batches.jsonl') for name in ['baseline','odb']}
    common = min(map(len, all_rows.values()))
    assert common > 0
    for a,b in zip(all_rows['baseline'][:common],all_rows['odb'][:common]):
        assert a['indices'] == b['indices'] and a['query_sha256'] == b['query_sha256'], 'Different PPO prompts'
    cases = {}
    probes = {}
    for name in ['baseline','odb']:
        root = output/name
        req = read(root/'REQUESTED_CONFIG.json')
        audit = read(root/'runs/short/PARAMETER_AUDIT.json')
        expected_sha = experiment['source_checkpoints'][name]['adapter_sha256']
        assert req['sft_adapter_sha256'] == audit['starting_adapter_sha256'] == expected_sha
        probes[name] = {stage: jsonl(root/f'probe_{stage}/rows.jsonl') for stage in ['before','after']}
        results = {stage: read(root/f'probe_{stage}/RESULT.json') for stage in ['before','after']}
        assert results['before']['indices'] == results['after']['indices']
        assert results['before']['adapter_sha256'] == expected_sha
        assert len(results['before']['indices']) == experiment['probe_samples']
        assert len(probes[name]['before']) == len(probes[name]['after']) == experiment['probe_samples']
        alert = root/'runs/short/QUALITY_STOPPED.json'
        cases[name] = dict(probe_before=results['before'], probe_after=results['after'],
            probe_accuracy_change_pp=100*(results['after']['numeric_accuracy']-results['before']['numeric_accuracy']),
            all_completed=phase_stats(all_rows[name]), common_prefix=phase_stats(all_rows[name][:common]),
            quality_stopped=read(alert) if alert.exists() else None,
            completed_requested_batches=len(all_rows[name]) == experiment['batches'],
            sft_adapter_sha256=expected_sha)
    pairs = {stage: paired_accuracy(probes['baseline'][stage],probes['odb'][stage]) for stage in ['before','after']}
    result = dict(experiment=experiment, common_prefix_batches=common, cases=cases, paired_probes=pairs,
        interpretation='Report observed before/after and control differences; no statistical guarantee of no quality loss. '
                       'Default 20 batches are entirely within the original 75-batch LR warmup. '
                       'K3 is relative to each case\'s own SFT reference; PPO loss alone is not a quality ranking.')
    lines = ['# ODB SFT → native PPO short test', '',
        f'Common PPO prefix: {common} batches; requested {experiment["batches"]}. '
        f'Probe: {experiment["probe_samples"]} shared training questions, excluded from the requested PPO prefix.', '',
        '| Metric | Ordinary SFT → PPO | ODB SFT → PPO |', '|---|---:|---:|']
    b,o=cases['baseline'],cases['odb']
    entries = [
        ('Before PPO numeric accuracy (%)',100*b['probe_before']['numeric_accuracy'],100*o['probe_before']['numeric_accuracy']),
        ('After PPO numeric accuracy (%)',100*b['probe_after']['numeric_accuracy'],100*o['probe_after']['numeric_accuracy']),
        ('Numeric accuracy change (percentage points)',b['probe_accuracy_change_pp'],o['probe_accuracy_change_pp']),
        ('Before PPO natural EOS (%)',100*b['probe_before']['natural_eos_rate'],100*o['probe_before']['natural_eos_rate']),
        ('After PPO natural EOS (%)',100*b['probe_after']['natural_eos_rate'],100*o['probe_after']['natural_eos_rate']),
        ('After PPO mean response tokens',b['probe_after']['mean_response_tokens'],o['probe_after']['mean_response_tokens'])]
    entries += [(label,b['common_prefix'][key],o['common_prefix'][key]) for label,key in [
        ('PPO common-prefix seconds','batch_wall_seconds'),('PPO seconds/batch','seconds_per_batch'),
        ('PPO input + response token/s','input_plus_response_tokens_per_second'),
        ('PPO response token/s','response_tokens_per_second'),('PPO reward fraction','reward_rate'),
        ('Mean K3','mean_k3'),('Max K3','max_k3'),('Mean PPO loss (diagnostic only)','mean_loss'),
        ('Peak allocated GiB','max_allocated_gib'),('OOM count','allocator_ooms')]]
    lines.extend(f'| {label} | {bv:.4f} | {ov:.4f} |' for label,bv,ov in entries)
    lines += ['', 'Paired probe discordances: ' + json.dumps(pairs), '',
        'Quality stops: ' + json.dumps({k:cases[k]['quality_stopped'] for k in cases}), '',
        result['interpretation'], '',
        'Timing uses sum of batch wall_seconds; separate state checkpoint saving, startup and probes are excluded. '
        'Token counts are actual input/response tokens, not multiplied by PPO epochs.', '',
        'Different SFT models produce different response lengths. A short pass does not establish held-out quality '
        'or full-training stability. Probe questions were used during SFT and are not a test-set evaluation.']
    write = lambda path,value: path.write_text(json.dumps(value,indent=2)+'\n')
    write(output/'RESULT.json',result)
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    print(summarize(args.experiment.resolve()))


if __name__ == '__main__':
    main()
