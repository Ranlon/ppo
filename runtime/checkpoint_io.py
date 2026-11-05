"""Complete state checkpoints for the isolated PPO loop, including exact resume RNG."""
from pathlib import Path
import copy
import dataclasses
import hashlib
import json
import os
import random
import shutil
import time
import numpy as np
import torch


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


def tensor_parameters(trainer):
    raw=trainer.accelerator.unwrap_model(trainer.model)
    return {name:cpu(param) for name,param in raw.named_parameters() if param.requires_grad}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def save_complete(trainer,force=False):
    step=trainer.state.global_step
    if step==0 or step==getattr(trainer,'_last_checkpoint_step',None):return
    if not force and step!=1 and step%trainer.state_save_every:return
    started=time.perf_counter()
    root=(trainer.output/'state_checkpoints').resolve();root.mkdir(exist_ok=True)
    target=root/f'step_{step:06d}'
    assert not target.exists(),str(target)
    target.mkdir()
    assert len(trainer.rows)==step and len(trainer.rollouts)==step*3
    raw=trainer.accelerator.unwrap_model(trainer.model)
    trainer.check_frozen_reference()
    scaler=getattr(trainer.accelerator,'scaler',None)
    running={k:cpu(v) for k,v in vars(trainer.running).items()
        if isinstance(v,(torch.Tensor,int,float,bool))} if hasattr(trainer,'running') else None
    state=dict(format_version=1,completed_batches=step,config_sha256=trainer.config_sha,
        policy_trainable=tensor_parameters(trainer),optimizer=cpu(trainer.optimizer.state_dict()),
        scheduler=cpu(trainer.lr_scheduler.state_dict()),kl_controller=cpu(vars(trainer.kl_ctl)),
        running_moments=running,scaler=cpu(scaler.state_dict()) if scaler is not None else None,
        trl_counters={k:cpu(getattr(trainer,k)) for k in ['current_step','compare_step','highest_reward'] if hasattr(trainer,k)},
        rng=rng_state(),sample_order=trainer.sample_order,
        trainer_state=dataclasses.asdict(trainer.state),batch_rows=trainer.rows,rollout_rows=trainer.rollouts,
        initial_trainable_sha256=trainer.initial_trainable_sha,
        prior_training_program_seconds=trainer.elapsed_before_resume+time.perf_counter()-trainer.program_start,
        frozen_sft_adapter_sha256=trainer.sft_sha,reward_revision=trainer.reward_revision,
        variant=trainer.variant,strict_determinism=trainer.strict,
        training_mode=raw.training,active_adapter=list(raw.pretrained_model.active_adapters),
        quality_guard=dict(zero_reward_streak=trainer.zero_reward_streak,quality_stop_reason=trainer.quality_stop_reason))
    partial=target/'training_state.pt.tmp';torch.save(state,partial)
    final=target/'training_state.pt';os.replace(partial,final)
    checksum=sha(final)
    record=dict(completed_batches=step,path=str(final),sha256=checksum,bytes=final.stat().st_size,
        complete_ppo_state=True,seconds=time.perf_counter()-started)
    (target/'MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
    pointer=trainer.output/'LATEST_CHECKPOINT.json';temporary=pointer.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,pointer)
    trainer._last_checkpoint_step=step
    with (trainer.output/'checkpoint_timings.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
    # Preserve all complete checkpoints for quality diagnosis and recovery.



def load_complete(trainer,path):
    path=Path(path).resolve()
    assert path.is_relative_to(trainer.root/'runs') and path.name=='training_state.pt'
    manifest=json.loads((path.parent/'MANIFEST.json').read_text())
    assert sha(path)==manifest['sha256']
    # Only load our own local state after its path and manifest are checked.
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['format_version']==1 and state['config_sha256']==trainer.config_sha
    assert state['sample_order']==trainer.sample_order and state['frozen_sft_adapter_sha256']==trainer.sft_sha
    assert state['reward_revision']==trainer.reward_revision and state['variant']==trainer.variant
    assert state['strict_determinism']==trainer.strict
    raw=trainer.accelerator.unwrap_model(trainer.model)
    actual={name:param for name,param in raw.named_parameters() if param.requires_grad}
    assert set(actual)==set(state['policy_trainable'])
    with torch.no_grad():
        for name,param in actual.items():param.copy_(state['policy_trainable'][name])
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.lr_scheduler.load_state_dict(state['scheduler'])
    from compare_resume import compare
    compare(cpu(trainer.optimizer.state_dict()),state['optimizer'],'loaded_optimizer')
    compare(cpu(trainer.lr_scheduler.state_dict()),state['scheduler'],'loaded_scheduler')
    for key,value in state['kl_controller'].items():setattr(trainer.kl_ctl,key,value)
    if state['running_moments']:
        for key,value in state['running_moments'].items():
            current=getattr(trainer.running,key)
            setattr(trainer.running,key,value.to(current.device) if isinstance(current,torch.Tensor) else value)
    if state['scaler'] is not None:trainer.accelerator.scaler.load_state_dict(state['scaler'])
    for key,value in state['trl_counters'].items():
        current=getattr(trainer,key)
        setattr(trainer,key,value.to(current.device) if isinstance(current,torch.Tensor) else value)
    for key,value in state['trainer_state'].items():setattr(trainer.state,key,value)
    trainer.rows=state['batch_rows'];trainer.rollouts=state['rollout_rows']
    trainer.initial_trainable_sha=state['initial_trainable_sha256']
    trainer.elapsed_before_resume=state['prior_training_program_seconds']+manifest['seconds']
    trainer._resume_step=state['completed_batches'];trainer._resume_rng=state['rng']
    trainer._last_checkpoint_step=state['completed_batches']
    trainer.zero_reward_streak=state['quality_guard']['zero_reward_streak']
    trainer.quality_stop_reason=state['quality_guard']['quality_stop_reason']
    for name,rows in [('batches.jsonl',trainer.rows),('rollouts.jsonl',trainer.rollouts)]:
        (trainer.output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    assert all(torch.equal(param.detach().cpu(),state['policy_trainable'][name]) for name,param in actual.items())
    trainer.check_frozen_reference()
    (trainer.output/'RESUME_AUDIT.json').write_text(json.dumps(dict(source=str(path),completed_batches=trainer._resume_step,
        complete_state_loaded=True,sample_order_matches=True,weights_bitwise_match_saved=True,
        optimizer_scheduler_kl_rng_present=True),indent=2)+'\n')
    return state
