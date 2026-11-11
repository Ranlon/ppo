"""Full-tuning configuration and content identities, isolated from LoRA runs."""
import json
from pathlib import Path
from project import PROJECT, read, sha, source_manifest, write


FULL_CODE = PROJECT / 'full_runtime'


def short_tmpdir():
    # multiprocess creates AF_UNIX sockets named pymp-*/listener-* below TMPDIR.
    # A timestamped run path leaves too little room for Linux's 108-byte sun_path.
    path = PROJECT / '.tmp'
    if len(str(path).encode()) > 65:
        raise ValueError(f'Project path too long for multiprocessing sockets: {path}')
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def build_manifest():
    result = source_manifest()
    result['run_full.py'] = sha(PROJECT / 'run_full.py')
    for path in sorted(FULL_CODE.glob('*.py')):
        result[path.relative_to(PROJECT).as_posix()] = sha(path)
    for path in sorted((PROJECT / 'tools').glob('*full*')):
        if path.is_file():
            result[path.relative_to(PROJECT).as_posix()] = sha(path)
    return result


def read_jsonl(path):
    # Unicode U+2028 inside a JSON string is NOT a JSONL record boundary.
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def full_model_identity(folder):
    folder = Path(folder)
    weights = sorted(folder.glob('*.safetensors')) or sorted(folder.glob('pytorch_model*.bin'))
    if not weights or not (folder / 'config.json').is_file():
        raise ValueError(f'Expected a full HF model, not a LoRA adapter: {folder}')
    if (folder / 'adapter_config.json').exists():
        raise ValueError(f'LoRA adapter is not a full model: {folder}')
    files = sorted(set(weights) | set(folder.glob('*.json')) | set(folder.glob('*.model')) |
                   set(folder.glob('*.tiktoken')) | set(folder.glob('*.jinja')) | set(folder.glob('*.txt')))
    hashes = {p.name: sha(p) for p in files}
    import hashlib
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(path=str(folder.resolve()), sha256=digest, files=hashes)


def make_profile(name, token_budget=1024, buffer_size=64, sft_lr=2e-5, ppo_lr=2e-7):
    if name not in ('baseline', 'optimized'):
        raise ValueError(name)
    if token_budget <= 0 or buffer_size < 2 or not (sft_lr > 0 and ppo_lr > 0):
        raise ValueError('Positive learning rates/token budget and buffer_size >= 2 required')
    # common.yaml is JSON; json.load avoids YAML's string scientific notation.
    common = read(PROJECT / 'configs/common.yaml')
    for stage in ('sft', 'ppo'):
        common[stage] = {k: v for k, v in common[stage].items() if not k.startswith('lora_')}
        common[stage].update(finetuning_type='full', pure_bf16=False,
                             learning_rate=float(sft_lr if stage == 'sft' else ppo_lr),
                             adam_epsilon=1e-8)
    # Full-parameter PPO is much more sensitive than LoRA. These settings
    # control the update/reference tradeoff; the K3=50 hard guard remains.
    common['full_ppo_protection'] = dict(learning_rate=float(ppo_lr),
        init_kl_coef=0.5, target=6.0, horizon=500, early_stopping=True, target_kl=1.0,
        rationale='Lower full-parameter update strength and stop oversized old/new PPO epochs')
    common.update(model='Qwen/Qwen3-0.6B-Base')
    common['trl_algorithm_contract'] = read(PROJECT / 'configs/trl_contract.json')
    return dict(name=name, experiment='full_06b', finetuning_type='full', common=common,
                sft=dict(checkpoint_policy='all', odb=(dict(token_budget=token_budget,
                    buffer_size=buffer_size, gradient_accumulation_steps=1) if name == 'optimized' else None)),
                runtime=dict(rollout='compiled_dense' if name == 'optimized' else 'native',
                    python_gc_per_batch=name == 'optimized',
                    head_probabilities='bf16_masked' if name == 'optimized' else 'native',
                    static_cache_length=768, lora_merge=False))


def attach_full_sft(root, folder):
    identity = full_model_identity(Path(folder).expanduser().resolve())
    request = read(root / 'REQUESTED_CONFIG.json')
    if (root / 'runs').exists() and any((root / 'runs').iterdir()):
        if request.get('sft_model_sha256') != identity['sha256']:
            raise ValueError('Cannot replace full SFT reference after PPO outputs were created')
    request.update(sft_checkpoint=identity['path'], sft_model_sha256=identity['sha256'],
                   sft_identity=identity)
    # Full PPO starts from SFT weights, never from the original base weights.
    request['requested_lf_arguments'].update(model_name_or_path=identity['path'],
                                           ref_model=identity['path'])
    write(root / 'REQUESTED_CONFIG.json', request)


def verify_full(root):
    assert read(root / 'FULL_RUNTIME_BUILD.json') == build_manifest(), 'Source changed; prepare a new full run'
    identity = read(root / 'INPUT_IDENTITY.json')
    base = Path(read(root / 'BASE_MODEL.json')['path'])
    for name, checksum in identity['model'].items():
        assert sha(base / name) == checksum, ('Base model changed', name)
    for name, item in identity['data'].items():
        assert sha(root / f'data/{name}.jsonl') == item['sha256']
    for name, checksum in identity['prepared_data'].items():
        assert sha(root / name) == checksum, ('Prepared data changed', name)
    request = read(root / 'REQUESTED_CONFIG.json')
    if request.get('sft_identity'):
        actual = full_model_identity(request['sft_checkpoint'])
        recorded = request['sft_identity']
        # /aistor and /hpc_stor01 can be two mount aliases for the same file.
        # Compare content identity, never the container-specific absolute path.
        assert actual['sha256'] == recorded['sha256'] and actual['files'] == recorded['files'], \
            'SFT model changed'


def tensor_hash(tensor):
    import hashlib
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def model_hashes(model):
    return {name: tensor_hash(param) for name, param in model.named_parameters()}


def restore_full_weights(model, folder):
    """Restore saved FP32 SFT values after LF's BF16-load/FP32-upcast path.

    Copy into existing storage, including tied parameters, without allocating a
    second GPU model. Full SFT outputs in this entry use safetensors.
    """
    import torch
    from safetensors import safe_open
    folder = Path(folder)
    targets = model.state_dict(keep_vars=True)
    expected_storage = {tensor.data_ptr() for tensor in targets.values()}
    restored_storage = set()
    shards = sorted(folder.glob('*.safetensors'))
    if not shards:
        raise ValueError(f'Expected this entry\'s full safetensors checkpoint: {folder}')
    for shard in shards:
        if shard.name == 'value_head.safetensors':
            continue
        with safe_open(shard, framework='pt', device='cpu') as handle, torch.no_grad():
            for name in handle.keys():
                if name not in targets:
                    raise ValueError(f'Unexpected full model tensor {name}')
                source = handle.get_tensor(name)
                target = targets[name]
                assert target.shape == source.shape, name
                if target.is_floating_point():
                    assert target.dtype == torch.float32, (name, target.dtype)
                target.copy_(source)
                restored_storage.add(target.data_ptr())
    assert restored_storage == expected_storage, 'Incomplete full model restore (including tied weights)'
