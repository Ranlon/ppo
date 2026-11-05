"""Explicit K3 KL estimator, detailed statistics and quality stop conditions."""
import math
import warnings
import torch

def k3(logp,logref):
    difference=(logref-logp).float()
    # No clipping of probability ratios: fail if numerical validity is lost.
    value=torch.expm1(difference)-difference
    if not torch.isfinite(value).all():raise FloatingPointError('Non-finite K3 KL')
    if value.min() < -1e-6:raise FloatingPointError('Negative K3 beyond roundoff')
    return value.clamp_min(0)

def initialize(trainer):
    warnings.filterwarnings('default',message='KL divergence is starting.*')
    warnings.filterwarnings('default',message='The average ratio of batch.*')
    trainer.raw_kl_sequences=[];trainer.zero_reward_streak=0;trainer.quality_stop_reason=None
    trainer.raw_masks=None

def penalty(trainer,logp,logref):
    trainer.raw_kl_sequences.append((logp-logref).detach())
    return k3(logp,logref)

def record(trainer,stats):
    assert len(trainer.raw_kl_sequences)==3 and trainer.raw_masks is not None
    signed=torch.stack(trainer.raw_kl_sequences)
    raw=float((signed*trainer.raw_masks).sum(-1).mean())
    coef=float(stats['objective/kl_coef']);objective=float(stats['objective/kl'])
    score=sum(trainer.pending['scores'])/3
    shaped=score-coef*objective
    assert objective>=-1e-6 and math.isfinite(shaped)
    assert score>0 or shaped<=1e-6,'Zero-score batch received positive KL reward'
    fields=dict(raw_signed_kl=raw,k3_kl=objective,kl_coef=coef,
        mean_summed_kl_reward=-coef*objective,mean_summed_shaped_reward=shaped,
        approxkl=float(stats['ppo/policy/approxkl']),policykl=float(stats['ppo/policy/policykl']),
        policy_clipfrac=float(stats['ppo/policy/clipfrac']),policy_loss=float(stats['ppo/loss/policy']),
        value_loss=float(stats['ppo/loss/value']),entropy=float(stats['ppo/policy/entropy']))
    assert all(math.isfinite(value) for value in fields.values())
    trainer.pending.update(fields)
    trainer.zero_reward_streak=trainer.zero_reward_streak+1 if score==0 else 0
    if trainer.zero_reward_streak>=20:trainer.quality_stop_reason='20 consecutive all-zero math-reward batches'
    if objective>50:trainer.quality_stop_reason=f'K3 KL exceeded safety bound: {objective}'
    recent=(getattr(trainer,'rows',[])+[trainer.pending])[-20:]
    if len(recent)==20 and all('eos_stopped' in row for row in recent):
        if sum(row['eos_stopped'] for row in recent)<30:
            trainer.quality_stop_reason='Natural EOS rate below 50% in the last 20 batches'
