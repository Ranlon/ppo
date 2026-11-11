"""Capture the legacy algorithm defaults actually used by the frozen trainer."""
from pathlib import Path
import json
from trl import PPOConfig

config=PPOConfig(batch_size=3,mini_batch_size=3,gradient_accumulation_steps=1,ppo_epochs=4,
    learning_rate=1e-6,max_grad_norm=1.,seed=42,target=6.,use_score_scaling=False,use_score_norm=False,whiten_rewards=False)
keys=['adap_kl_ctrl','init_kl_coef','kl_penalty','target','horizon','gamma','lam','cliprange',
      'cliprange_value','vf_coef','early_stopping','target_kl','ratio_threshold','score_clip',
      'use_score_scaling','use_score_norm','whiten_rewards','batch_size','mini_batch_size',
      'gradient_accumulation_steps','ppo_epochs','learning_rate','max_grad_norm','seed']
target=Path(__file__).resolve().parents[1]/'configs/trl_contract.json'
target.write_text(json.dumps({key:getattr(config,key) for key in keys},indent=2)+'\n',encoding='utf-8')
print(target.read_text())
