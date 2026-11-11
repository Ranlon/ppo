#!/usr/bin/env python3
"""Compare short native PPO from completed ordinary/ODB 4B LoRA SFT checkpoints."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choose_gpu(report, visible=None, requested=None, minimum=24576, max_util=10):
    allowed = None if visible is None else set(visible.split(','))
    candidates = []
    for fields in csv.reader(report.splitlines()):
        index, uuid, name, cc, free, util = (s.strip() for s in fields)
        if allowed is not None and index not in allowed and uuid not in allowed:
            continue
        if requested is not None and requested not in (index, uuid):
            continue
        if float(cc) >= 8 and int(free) >= minimum and int(util) <= max_util:
            candidates.append((int(free), -int(util), index, uuid, name))
    if not candidates:
        raise ValueError('No eligible visible GPU for this short test. Need >=24GiB free and <=10% utilization.\n' + report)
    return max(candidates)


def probe_indices(total, excluded, samples, seed=20261030):
    pool = sorted(set(range(total)) - set(excluded))
    if not 0 < samples <= len(pool):
        raise ValueError('Probe sample count exceeds questions outside the PPO prefix')
    return random.Random(seed).sample(pool, samples)


def snapshot_source(destination):
    """Do not edit/hash-bypass the live checkout or any existing training run."""
    destination.mkdir()
    shutil.copy2(PROJECT / 'run.py', destination / 'run.py')
    for folder in ['runtime', 'configs', 'vendor']:
        shutil.copytree(PROJECT / folder, destination / folder,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info', '.git'))
    path = destination / 'runtime/cli.py'
    original = path.read_text(encoding='utf-8')
    replacements = {
        "common=yaml.safe_load((PROJECT/'configs/common.yaml').read_text(encoding='utf-8'))":
        "common=json.loads((PROJECT/'configs/common.yaml').read_text(encoding='utf-8'))",
        "Path(path).read_text(encoding='utf-8-sig').splitlines()":
        "Path(path).open(encoding='utf-8-sig')",
        "json.dumps(r,ensure_ascii=False)+'\\n' for r in rows":
        "json.dumps(r,ensure_ascii=True)+'\\n' for r in rows",
    }
    patched = original
    for old, new in replacements.items():
        if old not in patched:
            raise ValueError('Snapshot source differs from reviewed obd_sft: ' + old)
        patched = patched.replace(old, new)
    path.write_text(patched, encoding='utf-8')
    write(destination / 'SNAPSHOT_PATCHES.json', dict(
        source=str(PROJECT), cli_before_sha256=hashlib.sha256(original.encode()).hexdigest(),
        cli_after_sha256=sha(path),
        fixes=['Parse JSON-format common config as JSON, keeping scientific notation numeric',
               'Read physical JSONL records, not Unicode splitlines',
               'Escape Unicode in prepared JSONL so existing reward/PPO readers remain compatible'],
        algorithm_changes=False, ppo_runtime='unchanged native baseline including all quality guards'))


def validate_sft(source, expect_odb):
    result = read(source / 'sft/SUCCESS.json')
    assert result['full_dataset_completed'] and not result['smoke']
    assert result['examples'] == 7473
    assert bool(result.get('odb')) == expect_odb, ('Unexpected SFT origin', source)
    adapter = source / 'sft/model'
    assert sha(adapter / 'adapter_model.safetensors') == result['adapter_sha256']
    config = read(adapter / 'adapter_config.json')
    assert config['r'] == 8 and config['lora_alpha'] == 16 and config['lora_dropout'] == 0
    request = read(source / 'REQUESTED_CONFIG.json')
    model = Path(request['requested_lf_arguments']['model_name_or_path'])
    arch = read(model / 'config.json')
    assert arch['model_type'] == 'qwen3' and arch['num_hidden_layers'] == 36 and arch['hidden_size'] == 2560
    return dict(source_run=str(source), adapter=str(adapter), adapter_sha256=result['adapter_sha256'],
                sft_optimizer_updates=result['optimizer_updates'], odb=result.get('odb'), model=str(model))


def execute(command, env, log):
    print('+', ' '.join(map(str, command)), flush=True)
    with log.open('x', encoding='utf-8') as handle:
        process = subprocess.Popen(list(map(str, command)), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pair = PROJECT / 'outputs/sft_odb_compare_20261105_175703'
    parser.add_argument('--baseline-sft-run', type=Path, default=pair / 'baseline')
    parser.add_argument('--odb-sft-run', type=Path, default=pair / 'odb')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--probe-samples', type=int, default=32)
    parser.add_argument('--gpu', default=os.environ.get('GPU'), help='Physical index or UUID, within visible devices')
    parser.add_argument('--prepare-only', action='store_true', help='CPU preparation only; no GPU probe or PPO')
    args = parser.parse_args()
    assert 1 <= args.batches <= 2491
    baseline = args.baseline_sft_run.expanduser().resolve()
    odb = args.odb_sft_run.expanduser().resolve()
    identities = dict(baseline=validate_sft(baseline, False), odb=validate_sft(odb, True))
    assert identities['baseline']['model'] == identities['odb']['model'], 'Base model paths differ'
    for split, count in [('train', 7473), ('test', 1319)]:
        a, b = jsonl(baseline / f'data/{split}.jsonl'), jsonl(odb / f'data/{split}.jsonl')
        assert len(a) == len(b) == count and a == b, 'Source SFT data differ'
    import torch
    order = torch.randperm(7473, generator=torch.Generator().manual_seed(42)).tolist()
    indices = probe_indices(7473, order[:args.batches*3], args.probe_samples)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = (args.output or PROJECT / f'outputs/odb_ppo_smoke_{stamp}').expanduser().resolve()
    if not output.is_relative_to(PROJECT):
        raise ValueError('Keep this experiment output inside Project200/ppo')
    if output.exists():
        raise FileExistsError(f'Choose a new output: {output}')
    env = os.environ.copy()
    env.update(DISABLE_VERSION_CHECK='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               WANDB_DISABLED='true', OMP_NUM_THREADS='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    gpu = None
    if not args.prepare_only:
        report = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,compute_cap,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        free, neg_util, index, uuid, name = choose_gpu(report, env.get('CUDA_VISIBLE_DEVICES'), args.gpu)
        env['CUDA_VISIBLE_DEVICES'] = uuid
        gpu = dict(index=index, uuid=uuid, name=name, free_mib=free, utilization=-neg_util)
        print('Selected GPU:', gpu, flush=True)
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    output.mkdir(parents=True)
    tmp = PROJECT / '.tmp'
    tmp.mkdir(mode=0o700, exist_ok=True)
    assert len(str(tmp).encode()) <= 65, 'Project tmp path too long for AF_UNIX sockets'
    env.update(TMPDIR=str(tmp), HF_HOME=str(output / 'hf_cache'), XDG_CACHE_HOME=str(output / 'cache'))
    source = output / 'source'
    snapshot_source(source)
    write(output / 'EXPERIMENT.json', dict(source_checkpoints=identities, gpu=gpu,
        ppo_profile='baseline', rollout='native', batches=args.batches, ppo_epochs=4,
        strict=False, scheduler='original 2491 batch schedule; 75 batch warmup is NOT shortened',
        probe_samples=args.probe_samples, prepare_only=args.prepare_only,
        source_snapshot=str(source), controls='Same code, input order, seeds, batch3, PPO4, reward and guards; only SFT differs',
        limitation='Short training-question diagnostics cannot establish held-out noninferiority or long-run stability'))
    for name in ['baseline', 'odb']:
        root = output / name
        cmd = [sys.executable, source / 'run.py', 'prepare', '--profile', 'baseline', '--run-dir', root,
               '--model', identities[name]['model'], '--train', baseline / 'data/train.jsonl',
               '--test', baseline / 'data/test.jsonl']
        if execute(cmd, env, output / f'{name}_prepare.log'):
            raise RuntimeError(f'{name} prepare failed')
        # Validate and attach existing full-epoch adapter without running SFT again.
        code = ('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                'from cli import attach_sft,verify_run;root=Path(sys.argv[2]);'
                'verify_run(root);attach_sft(root,sys.argv[3])')
        if execute([sys.executable, '-c', code, source / 'runtime', root, identities[name]['adapter']],
                   env, output / f'{name}_attach.log'):
            raise RuntimeError(f'{name} adapter attach failed')
    train_sha = sha(output / 'baseline/data/train.jsonl')
    assert train_sha == sha(output / 'odb/data/train.jsonl')
    write(output / 'PROBE_PLAN.json', dict(indices=indices, train_sha256=train_sha,
        excluded_ppo_prefix_examples=args.batches*3, seed=20261030,
        training_questions_only=True, no_test_questions_used=True))
    print('Experiment:', output, flush=True)
    if args.prepare_only:
        print('Preparation complete. No GPU probe or PPO was executed.'); return
    for name in ['baseline', 'odb']:
        root = output / name
        for stage in ['before', 'ppo', 'after']:
            subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                '--format=csv'], stdout=(output / f'{name}_{stage}_gpu.csv').open('w'), check=True)
            if stage == 'ppo':
                cmd = [sys.executable, source / 'run.py', 'ppo', '--run-dir', root,
                       '--name', 'short', '--limit', args.batches, '--save-every', 250]
                status = execute(cmd, env, root / 'ppo.stdout.log')
                if status and not (root / 'runs/short/QUALITY_STOPPED.json').exists():
                    raise RuntimeError(f'{name} PPO failed; inspect {root}/ppo.stdout.log')
            else:
                adapter = Path(identities[name]['adapter']) if stage == 'before' else root / 'runs/short/model'
                cmd = [sys.executable, PROJECT / 'tools/odb_ppo_probe.py', '--source', source,
                       '--run-dir', root, '--adapter', adapter, '--plan', output / 'PROBE_PLAN.json',
                       '--output', root / f'probe_{stage}']
                if execute(cmd, env, root / f'probe_{stage}.stdout.log'):
                    raise RuntimeError(f'{name} {stage} probe failed')
    for name in ['baseline', 'odb']:
        if sha(Path(identities[name]['adapter']) / 'adapter_model.safetensors') != identities[name]['adapter_sha256']:
            raise RuntimeError('Original SFT checkpoint changed')
    subprocess.run([sys.executable, PROJECT / 'tools/odb_ppo_summary.py', '--experiment', output], check=True)


if __name__ == '__main__':
    main()
