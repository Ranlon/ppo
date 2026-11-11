"""Prepare and launch independent full-parameter experiments; never start on import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from project import PROJECT, CODE, VENDOR, ODB_VENDOR, read, write, sha
from full_support import (make_profile, read_jsonl, full_model_identity, attach_full_sft,
                          build_manifest, verify_full, FULL_CODE, short_tmpdir)


def prepare(args):
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.unsupervised import UnsupervisedDatasetProcessor
    from numeric_reward import numeric_answer

    selected = make_profile(args.profile, args.odb_token_budget, args.odb_buffer_size,
                            args.sft_lr, args.ppo_lr)
    model = Path(args.model).expanduser().resolve()
    config = read(model / 'config.json')
    assert config['model_type'] == 'qwen3' and config['num_hidden_layers'] == 28
    assert config['hidden_size'] == 1024 and config['tie_word_embeddings']
    assert not config.get('quantization_config'), 'Full training requires unquantized weights'
    root = Path(args.run_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f'Choose a new run directory: {root}')
    rows = {}
    for name, source, count in [('train', args.train, 7473), ('test', args.test, 1319)]:
        if Path(source).suffix == '.parquet':
            import pyarrow.parquet as pq
            values = pq.read_table(source, columns=['question', 'answer']).to_pylist()
        else:
            values = read_jsonl(source)
        assert len(values) == count, (name, len(values))
        assert len({r['question'] for r in values}) == count
        assert all(isinstance(r['question'], str) and r['question'].strip() and
                   isinstance(r['answer'], str) and numeric_answer(r['answer'], gold=True) is not None
                   for r in values)
        rows[name] = values
    assert not ({r['question'] for r in rows['train']} & {r['question'] for r in rows['test']})
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    data_args = DataArguments(template='qwen3_nothink', cutoff_len=1024)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = UnsupervisedDatasetProcessor(template=template, tokenizer=tokenizer,
                                             processor=None, data_args=data_args)
    metadata = []
    for index, row in enumerate(rows['train']):
        ids, _ = processor._encode_data_example([dict(role='user', content=row['question'])],
                                                 [], None, None, [], [], [])
        metadata.append(dict(index=index, prompt_tokens=len(ids), prompt_token_sha256=hashlib.sha256(
            json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
    maximum = max(r['prompt_tokens'] for r in metadata)
    assert maximum + 512 <= selected['runtime']['static_cache_length'], 'StaticCache too short'
    identity = full_model_identity(model)
    root.mkdir(parents=True)
    data = root / 'data'
    data.mkdir()
    manifest = {}
    for name, values in rows.items():
        path = data / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=True) + '\n' for r in values), encoding='utf-8')
        manifest[name] = dict(sha256=sha(path), examples=len(values))
    for name, output in [('gsm8k_sft', True), ('gsm8k_ppo', False)]:
        write(data / f'{name}.json', [dict(instruction=r['question'], input='',
            output=r['answer'] if output else '') for r in rows['train']])
    write(data / 'dataset_info.json', {name: dict(file_name=f'{name}.json',
          columns=dict(prompt='instruction', query='input', response='output'))
          for name in ('gsm8k_sft', 'gsm8k_ppo')})
    write(data / 'UPSTREAM_MANIFEST.json', manifest)
    (root / 'PPO_PROMPT_METADATA.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metadata))
    derived = {p.relative_to(root).as_posix(): sha(p) for p in [data / 'gsm8k_sft.json',
        data / 'gsm8k_ppo.json', data / 'dataset_info.json', root / 'PPO_PROMPT_METADATA.jsonl']}
    write(root / 'INPUT_IDENTITY.json', dict(model=identity['files'], data=manifest, prepared_data=derived))
    write(root / 'BASE_MODEL.json', identity)
    reward = dict(model='GSM8K exact numeric answer rule (no learned reward model)',
        reward_type='exact_numeric_answer', correct_score=1., incorrect_score=0.,
        train_sha256=manifest['train']['sha256'], scorer_source_sha256=sha(CODE / 'numeric_reward.py'),
        training_labels_only=True, learned_reward_model_loaded=False, format_bonus=False)
    reward['revision'] = hashlib.sha256(json.dumps(reward, sort_keys=True).encode()).hexdigest()
    write(root / 'REWARD_SPEC.json', reward)
    sft = dict(selected['common']['sft'], model_name_or_path=str(model), dataset_dir=str(data),
               output_dir=str(root / 'sft/model'), cache_dir=str(root / 'cache_sft'))
    ppo = dict(selected['common']['ppo'], model_name_or_path=str(model), dataset_dir=str(data),
        output_dir=str(root / 'runs/UNSTARTED/model'), cache_dir=str(root / 'cache'),
        ppo_batch_python_gc=selected['runtime']['python_gc_per_batch'],
        ppo_compiled_rollout=selected['runtime']['rollout'] == 'compiled_dense')
    write(root / 'SFT_CONFIG.json', sft)
    write(root / 'REQUESTED_CONFIG.json', dict(finetuning_type='full', model='Qwen3-0.6B-Base',
        requested_lf_arguments=ppo, sft_checkpoint=None, sft_model_sha256=None, reward=reward,
        repair=dict(kl_estimator='K3', k3_stop=50, quality_zero_batch_stop=20),
        raw_training_examples=7473, rollout_batches=2491, optimizer_updates=9964))
    write(root / 'PROFILE.json', selected)
    write(root / 'DATA_PREFLIGHT.json', dict(passed=True, train_examples=7473, test_examples=1319,
          max_ppo_prompt_tokens=maximum, formal_max_new_tokens=512))
    write(root / 'FULL_RUNTIME_BUILD.json', build_manifest())
    print(json.dumps(dict(prepared=str(root), profile=args.profile, finetuning_type='full',
                     max_ppo_prompt_tokens=maximum), indent=2))


def launch(root, script, *arguments):
    env = os.environ.copy()
    env.update(PPO4_RUN_ROOT=str(root), PYTHONPATH=os.pathsep.join(map(str, [FULL_CODE, CODE, VENDOR, ODB_VENDOR])),
        DISABLE_VERSION_CHECK='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true',
        TMPDIR=str(short_tmpdir()))
    folder = FULL_CODE if script.startswith('full_') else CODE
    subprocess.run([sys.executable, str(folder / script), *map(str, arguments)], env=env, check=True)


def run_ppo(root, args):
    from urllib.request import urlopen
    selected = read(root / 'PROFILE.json')
    name = args.name
    assert name and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name)
    assert not (root / 'runs' / name).exists(), 'PPO output exists; choose a new --name'
    if args.sft:
        attach_full_sft(root, args.sft)
    request = read(root / 'REQUESTED_CONFIG.json')
    assert request['sft_checkpoint'], 'Complete this run\'s full SFT first, or supply --sft'
    verify_full(root)
    # Full model identity was checked above. Do not delegate to the LoRA CLI,
    # which requires adapter_model.safetensors and an adapter-only SHA.
    ready = root / f'REWARD_SERVER_{name}.json'
    assert not ready.exists(), 'Reward-server record exists; choose a new --name'
    env = os.environ.copy()
    env.update(PPO4_RUN_ROOT=str(root), PYTHONPATH=os.pathsep.join(map(str, [CODE, VENDOR, ODB_VENDOR])),
               TMPDIR=str(short_tmpdir()))
    with (root / f'reward_{name}.stdout.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(CODE / 'reward_server.py'), '--name', name],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while not ready.exists():
                if child.poll() is not None:
                    raise RuntimeError(f'Reward server exited; inspect {log.name}')
                if time.monotonic() > deadline:
                    raise TimeoutError('Reward server readiness timeout')
                time.sleep(.1)
            url = read(ready)['url']
            with urlopen(url.replace('/reward', '/'), timeout=5) as response:
                assert json.load(response)['ready']
            extra = ['--stress'] if args.stress else []
            if args.resume_state:
                extra += ['--resume-state', str(Path(args.resume_state).resolve())]
            if args.strict:
                extra += ['--strict']
            launch(root, 'full_train_ppo.py', '--name', name, '--variant',
                   'native' if selected['runtime']['rollout'] == 'native' else 'compiled',
                   '--reward-url', url, '--limit', args.limit,
                   '--state-save-every', args.save_every, *extra)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()


def main():
    os.environ.setdefault('DISABLE_VERSION_CHECK', '1')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('config')
    c.add_argument('--profile', choices=['baseline', 'optimized'], default='baseline')
    d = sub.add_parser('doctor')
    d.add_argument('--gpu', action='store_true')
    p = sub.add_parser('prepare')
    p.add_argument('--profile', choices=['baseline', 'optimized'], required=True)
    p.add_argument('--run-dir', required=True)
    for key in ('model', 'train', 'test'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--odb-token-budget', type=int, default=1024)
    p.add_argument('--odb-buffer-size', type=int, default=64)
    p.add_argument('--sft-lr', type=float, default=2e-5)
    p.add_argument('--ppo-lr', type=float, default=2e-7)
    for cmd in ('sft', 'ppo', 'all', 'verify', 'evaluate'):
        item = sub.add_parser(cmd)
        item.add_argument('--run-dir', required=True)
        if cmd in ('sft', 'all'):
            item.add_argument('--sft-limit', type=int, default=935,
                              help='Native full SFT updates; ODB full epoch uses sample coverage')
            item.add_argument('--sft-name', default='sft')
            item.add_argument('--resume-hf')
        if cmd in ('ppo', 'all'):
            item.add_argument('--name', default='full')
            item.add_argument('--limit', type=int, default=2491)
            item.add_argument('--save-every', type=int, default=250)
            item.add_argument('--sft', help='Completed full SFT model directory, not an adapter')
            item.add_argument('--resume-state')
            item.add_argument('--strict', action='store_true')
            item.add_argument('--stress', action='store_true')
        if cmd == 'evaluate':
            item.add_argument('--model', required=True)
            item.add_argument('--label', choices=['SFT', 'PPO'], required=True)
    args = parser.parse_args()
    if args.command == 'config':
        print(json.dumps(make_profile(args.profile), indent=2)); return
    if args.command == 'doctor':
        from doctor import main as doctor
        doctor(args.gpu); return
    if args.command == 'prepare':
        prepare(args); return
    root = Path(args.run_dir).expanduser().resolve()
    verify_full(root)
    if args.command == 'verify':
        print(json.dumps(dict(passed=True, run_dir=str(root)))); return
    if args.command in ('sft', 'all'):
        assert 0 < args.sft_limit <= 935
        extra = ['--resume-hf', args.resume_hf] if args.resume_hf else []
        launch(root, 'full_train_sft.py', '--limit', args.sft_limit, '--name', args.sft_name, *extra)
        if args.sft_limit == 935:
            attach_full_sft(root, root / args.sft_name / 'model')
        if args.command == 'sft': return
        assert args.sft_limit == 935 and not args.stress, 'all requires complete SFT'
    if args.command in ('ppo', 'all'):
        assert 0 < args.limit <= 2491 and args.save_every > 0
        run_ppo(root, args); return
    if args.command == 'evaluate':
        launch(root, 'full_evaluate.py', '--model', args.model, '--label', args.label)
