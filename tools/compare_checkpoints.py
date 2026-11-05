"""Compare algorithm state of two trusted local complete PPO checkpoints."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import torch
from compare_resume import compare
from project import read,sha

def load(path):
    path=Path(path).resolve()
    assert path.name=='training_state.pt'
    assert read(path.parent/'MANIFEST.json')['sha256']==sha(path)
    return torch.load(path,map_location='cpu',weights_only=False)

parser=argparse.ArgumentParser();parser.add_argument('--control',required=True);parser.add_argument('--resumed',required=True)
args=parser.parse_args();a=load(args.control);b=load(args.resumed)
keys=['format_version','completed_batches','config_sha256','policy_trainable','optimizer','scheduler',
      'kl_controller','running_moments','scaler','trl_counters','rng','sample_order',
      'initial_trainable_sha256','frozen_sft_adapter_sha256','reward_revision','variant','strict_determinism','quality_guard']
for key in keys:compare(a[key],b[key],key)
print(json.dumps(dict(passed=True,algorithm_state_bitwise_equal=True,completed_batches=a['completed_batches'],
    excluded='wall times, compile state, training/throughput logs')))
