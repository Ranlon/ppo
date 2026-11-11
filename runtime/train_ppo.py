"""Fresh GSM8K-SFT PPO with a frozen SFT reference, exact numeric rule rewards and full resume."""
import os
import sys
from pathlib import Path
os.environ['DISABLE_VERSION_CHECK']='1'
os.environ['ACCELERATE_MIXED_PRECISION']='bf16'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['WANDB_DISABLED']='true'
if '--strict' in sys.argv:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from project import run_root,CODE,VENDOR
ROOT=run_root()
sys.path.insert(0,str(VENDOR))
os.environ['PYTHONPATH']=str(CODE)+os.pathsep+str(VENDOR)
_cache_scope='full_cold' if '--name' in sys.argv and sys.argv[sys.argv.index('--name')+1]=='full' else 'validation'
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'compiler_cache'/_cache_scope/'inductor')
os.environ['TRITON_CACHE_DIR']=str(ROOT/'compiler_cache'/_cache_scope/'triton')


def main():
    if 'LOCAL_RANK' not in os.environ:
        os.execv(sys.executable,[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(Path(__file__).resolve()),*sys.argv[1:]])
    import argparse
    import collections
    import functools
    import hashlib
    import importlib.metadata as metadata
    import json
    import math
    import re
    import signal
    import time
    import requests
    import torch
    from torch.utils.data import DataLoader
    from transformers import TrainerCallback,set_seed
    from adapter_io import install
    install()
    import repair_policy
    from peft import get_peft_model_state_dict
    from llamafactory.hparams import get_train_args
    from llamafactory.train.ppo import workflow
    from frozen_sft_reference import FrozenSFTReference,restore_exact_adapter
    from runtime_cleanup import RuntimeCleanupAudit
    from checkpoint_io import load_complete,save_complete,restore_rng,sha
    from entropy_stats import install as install_chunked_entropy
    install_chunked_entropy()
    from logprob_memory import install as install_chunked_logprob
    install_chunked_logprob()

    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--variant',choices=['native','compiled'],required=True)
    parser.add_argument('--reward-url',required=True)
    parser.add_argument('--limit',type=int,default=2491)
    parser.add_argument('--state-save-every',type=int,default=250)
    parser.add_argument('--resume-state')
    parser.add_argument('--strict',action='store_true')
    parser.add_argument('--stress',action='store_true')
    args=parser.parse_args()
    assert re.fullmatch('[a-z0-9_-]+',args.name)
    from project import runtime_options,PROJECT
    options=runtime_options()
    args.head_mode='baseline' if options['head_probabilities']=='native' else 'bf16_masked'
    args.head_chunk=32;args.checkpoint_policy='all';args.trunk_policy='padded'
    assert args.reward_url.startswith('http://127.0.0.1:') and 0<args.limit<=2491
    assert metadata.version('llamafactory').split('+')[0]=='0.9.4' and metadata.version('trl')=='0.9.6'
    assert metadata.version('transformers')=='4.57.1' and metadata.version('peft')=='0.17.1'
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    if args.strict:
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.set_autocast_dtype('cuda',torch.bfloat16)
    set_seed(42)
    output=ROOT/'runs'/args.name;output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    def write(path,value):
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
    def tensor_sha(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    request=read(ROOT/'REQUESTED_CONFIG.json')
    raw_rows=[json.loads(line) for line in (ROOT/'data/train.jsonl').read_text().splitlines()]
    metadata_rows=[json.loads(s) for s in (ROOT/'PPO_PROMPT_METADATA.jsonl').read_text().splitlines()]
    reward=read(ROOT/'REWARD_SPEC.json')
    config=dict(request['requested_lf_arguments'])
    config.update(dataset='gsm8k_ppo',dataset_dir=str(ROOT/'data'),reward_model=args.reward_url,
        output_dir=str(output/'model'),cache_dir=str(ROOT/'cache'),plot_loss=args.limit==2491,
        ppo_batch_python_gc=bool(config['ppo_batch_python_gc']),ppo_compiled_rollout=args.variant=='compiled')
    config.pop('max_steps',None);config.pop('resume_from_checkpoint',None)
    assert config['ppo_epochs']==4 and config['per_device_train_batch_size']==3
    assert config['learning_rate']==1e-6 and config['top_k']==0 and config['top_p']==config['temperature']==1
    assert (args.variant=='compiled')==bool(request['requested_lf_arguments']['ppo_compiled_rollout'])
    algorithm_config={k:v for k,v in config.items() if k not in ['model_name_or_path','adapter_name_or_path','dataset_dir','output_dir','cache_dir','reward_model','plot_loss']}
    algorithm_config['portable_runtime']=options
    algorithm_config['external_content']=read(ROOT/'INPUT_IDENTITY.json')
    config_sha=hashlib.sha256(json.dumps(dict(config=algorithm_config,runtime=read(ROOT/'RUNTIME_BUILD.json'),
        sft=request['sft_adapter_sha256'],reward_revision=reward['revision'],strict=args.strict,
        gradient_clip_parameter_iterator='materialized_trainable_tuple_v1',
        entropy_statistics='native_formula_no_grad_token_chunks32_v1',
        selected_logprobs='native_cuda_forward_backward_token_chunks32_v1',repair=request['repair'],
        repair_sources={name:sha(CODE/name) for name in ['repair_policy.py','adapter_io.py']}),sort_keys=True).encode()).hexdigest()
    write(output/'RUN_CONFIG.json',config);write(output/'CLI.json',vars(args))
    model_holder=[];trainer_holder=[]
    native_loader=workflow.load_model
    def load_model(*a,**kw):
        model=native_loader(*a,**kw)
        restore_exact_adapter(model.pretrained_model,request['sft_checkpoint'],'default')
        ref=FrozenSFTReference(model.pretrained_model,request['sft_checkpoint'])
        model_holder.append((model,ref))
        return model
    workflow.load_model=load_model
    cleanup=RuntimeCleanupAudit('native',torch,native_python_gc=config['ppo_batch_python_gc'],expected_cache_calls=3+2*config['ppo_epochs'])
    native_class=workflow.CustomPPOTrainer

    class AuditCallback(TrainerCallback):
        def on_step_end(self,training_args,state,control,**kw):
            if state.global_step>=args.limit:control.should_training_stop=True
            trainer=trainer_holder[0]
            if trainer.quality_stop_reason:
                write(output/'QUALITY_ALERT.json',dict(batch=state.global_step,reason=trainer.quality_stop_reason))
                control.should_training_stop=True
            return control
        def on_train_end(self,training_args,state,control,**kw):
            trainer=trainer_holder[0];trainer.finish_batch();save_complete(trainer,force=True)

    class GSM8KPPOTrainer(native_class):
        def __init__(self,*a,**kw):
            collator=kw['data_collator']
            super().__init__(*a,**kw)
            # TRL 0.9.6 stores a one-shot filter here. Every clipping call must see all parameters.
            self.model_params=tuple(self.model_params)
            for key,value in read(PROJECT/'configs/trl_contract.json').items():
                assert getattr(self.config,key)==value,(key,getattr(self.config,key),value)
            assert sum(p.numel() for p in self.model_params)==17753601
            self.root=ROOT.resolve();self.output=output;self.program_start=started;self.elapsed_before_resume=0.0
            self.variant=args.variant;self.strict=args.strict;self.config_sha=config_sha
            self.state_save_every=args.state_save_every;self.sft_sha=request['sft_adapter_sha256'];self.reward_revision=reward['revision']
            self.reference=model_holder[0][1]
            self.optional_peft_ctx=self.reference.context
            self.amp_context=torch.autocast('cuda',dtype=torch.bfloat16)
            assert self.tokenizer.eos_token_id==151645
            self.generation_config.eos_token_id=[self.tokenizer.eos_token_id]
            self.generation_config.min_length=0
            self.generation_config.min_new_tokens=0
            assert self.config.batch_size==self.config.mini_batch_size==3 and self.config.ppo_epochs==config['ppo_epochs']==4
            assert len(self.dataset)==7473 and len(self.dataloader)==2491
            assert self.args.get_warmup_steps(2491)==75 and self.args.max_steps==-1
            assert self.accelerator.mixed_precision=='bf16' and self.accelerator.num_processes==1
            for index,ids in enumerate(self.dataset['input_ids']):
                digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
                assert digest==metadata_rows[index]['prompt_token_sha256'],index
            self.sample_order=torch.randperm(7473,generator=torch.Generator().manual_seed(42)).tolist()
            ordered=self.dataset.select(self.sample_order)
            self.dataloader=self.accelerator.prepare_data_loader(DataLoader(ordered,batch_size=3,shuffle=False,
                collate_fn=collator,drop_last=False,num_workers=0,pin_memory=True))
            assert len(self.dataloader)==2491
            self.longest=sorted(range(7473),key=lambda i:len(self.dataset[i]['input_ids']),reverse=True)[:3]
            self.rows=[];self.rollouts=[];self.pending=None;self._resume_step=0;self._resume_rng=None
            self.calls=collections.Counter();self.forward_calls=collections.Counter();self.no_grad_pass=0
            self.raw=self.accelerator.unwrap_model(self.model)
            self.reference_hash={k:tensor_sha(v) for k,v in get_peft_model_state_dict(self.raw.pretrained_model,adapter_name='sft_reference').items()}
            trainable={name:param for name,param in self.raw.named_parameters() if param.requires_grad}
            assert sum(param.numel() for param in trainable.values())==17753601
            assert all(('lora_' in name and '.default.' in name) or name.startswith('v_head.') for name in trainable)
            assert all(param.dtype==torch.float32 for param in trainable.values())
            self.initial_trainable_sha={name:tensor_sha(param) for name,param in trainable.items()}
            for obj,method,label in [(self.optimizer,'step','optimizer'),(self.accelerator,'backward','backward')]:
                original=getattr(obj,method)
                def make_counted(fn,key):
                    @functools.wraps(fn)
                    def counted(*aa,**kk):
                        self.calls[key]+=1
                        if args.strict and key=='optimizer':
                            audit=dict(batch=self.state.global_step+1,call=self.calls[key],
                                gradients={n:tensor_sha(p.grad) for n,p in self.raw.named_parameters() if p.grad is not None},
                                parameters={n:tensor_sha(p) for n,p in self.raw.named_parameters() if p.requires_grad})
                        value=fn(*aa,**kk)
                        if args.strict and key=='optimizer':
                            audit['parameters_after']={n:tensor_sha(p) for n,p in self.raw.named_parameters() if p.requires_grad}
                            with (output/'UPDATE_AUDIT.jsonl').open('a') as handle:handle.write(json.dumps(audit)+'\n')
                        return value
                    return counted
                setattr(obj,method,make_counted(original,label))
            repair_policy.initialize(self)
            if args.resume_state:load_complete(self,args.resume_state)
            write(output/'PARAMETER_AUDIT.json',dict(examples=7473,batches=2491,warmup_batches=75,
                ppo_epochs=config['ppo_epochs'],batch_size=3,trainable_parameters=17753601,fp32_lora_parameters=17751040,
                value_head_parameters=2561,reference_adapter_frozen=True,reference_policy='GSM8K SFT',
                starting_adapter_sha256=self.sft_sha,initial_trainable_sha256=self.initial_trainable_sha,variant=args.variant,sample_order_sha256=hashlib.sha256(json.dumps(self.sample_order).encode()).hexdigest(),
                strict_determinism=args.strict,compiled_base_backup_bytes='See migration_runtime ledger',repeatable_gradient_clipping_parameters=True,
                entropy_statistics='Native formula, no_grad, 32-token chunks; only logging statistics',
                selected_logprobs='Native log_softmax and native backward in 32-token chunks, no approximation',
                versions={name:metadata.version(name) for name in ['llamafactory','trl','torch','peft','transformers','accelerate']}))
            self._m9_head_config=dict(mode=args.head_mode,chunk=args.head_chunk,checkpoint=args.checkpoint_policy,trunk=args.trunk_policy)
            from head_runtime import install_entropy_dispatch
            install_entropy_dispatch()
            trainer_holder.append(self)

        def check_frozen_reference(self):
            actual={k:tensor_sha(v) for k,v in get_peft_model_state_dict(self.raw.pretrained_model,adapter_name='sft_reference').items()}
            assert actual==self.reference_hash
            assert list(self.raw.pretrained_model.active_adapters)==['default']
            assert all(not p.requires_grad for n,p in self.raw.named_parameters() if '.sft_reference.' in n)

        def _apply_runtime_rng(self):restore_rng(self._resume_rng)

        def get_inputs(self,batch):
            self.finish_batch();save_complete(self)
            step=self.state.global_step+1
            indices=self.sample_order[(step-1)*3:step*3]
            if args.stress:
                indices=self.longest
                previous=self.tokenizer.padding_side;self.tokenizer.padding_side='left'
                try:batch=self.tokenizer.pad([dict(input_ids=self.dataset[i]['input_ids'],attention_mask=[1]*len(self.dataset[i]['input_ids'])) for i in indices],padding=True,return_tensors='pt').to(self.current_device)
                finally:self.tokenizer.padding_side=previous
                self.generation_config.min_new_tokens=512
            self.calls.clear();self.forward_calls.clear();self.no_grad_pass=0
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            self.pending=dict(batch=step,indices=indices,started=time.perf_counter())
            self.raw_kl_sequences=[]
            cleanup.start()
            phase_started=time.perf_counter()
            queries,responses=super().get_inputs(batch)
            torch.cuda.synchronize()
            self.pending['generation_seconds']=time.perf_counter()-phase_started
            for index,query in zip(indices,queries):
                assert hashlib.sha256(json.dumps(query.tolist(),separators=(',',':')).encode()).hexdigest()==metadata_rows[index]['prompt_token_sha256']
            self.pending.update(input_tokens=sum(len(q) for q in queries),response_tokens=sum(len(r) for r in responses),
                eos_stopped=sum(int(r[-1])==self.tokenizer.eos_token_id for r in responses),
                token_cap_responses=sum(len(r)==512 and int(r[-1])!=self.tokenizer.eos_token_id for r in responses),
                response_lengths=[len(r) for r in responses],query_sha256=hashlib.sha256(b''.join(q.numpy().tobytes() for q in queries)).hexdigest(),
                response_sha256=hashlib.sha256(b''.join(r.numpy().tobytes() for r in responses)).hexdigest())
            return queries,responses

        @torch.no_grad()
        def get_rewards(self,queries,responses):
            indices=self.pending['indices']
            prompts=[raw_rows[i]['question'] for i in indices]
            texts=self.tokenizer.batch_decode(responses,skip_special_tokens=True)
            response=requests.post(args.reward_url,json=dict(prompts=prompts,responses=texts,indices=indices,run=args.name),timeout=180)
            response.raise_for_status();data=response.json()
            assert len(data['scores'])==3 and all(math.isfinite(value) for value in data['scores'])
            self.pending.update(scores=data['scores'],reward_seconds=data['seconds'],reward_input_tokens=data['input_tokens'])
            for i,text,score in zip(indices,texts,data['scores']):
                record=dict(batch=self.pending['batch'],index=i,response=text,reward=score,
                    strict_format_present='####' in text,empty=not text.strip())
                self.rollouts.append(record)
                with (output/'rollouts.jsonl').open('a') as handle:handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            return [torch.tensor(value,device=self.current_device) for value in data['scores']]

        def batched_forward_pass(self,*a,**kw):
            phase='train_forward' if torch.is_grad_enabled() else 'old_policy' if self.no_grad_pass==0 else 'reference'
            if not torch.is_grad_enabled():self.no_grad_pass+=1
            self.forward_calls[phase]+=1
            if args.strict:
                audit=dict(batch=self.state.global_step+1,phase=phase,call=self.forward_calls[phase],
                    rng=tensor_sha(torch.cuda.get_rng_state()),
                    training={n:m.training for n,m in self.raw.named_modules() if isinstance(m,torch.nn.Dropout)})
            if args.head_mode=='baseline':
                from head_runtime import apply_checkpoint_policy
                apply_checkpoint_policy(self.raw,args.checkpoint_policy)
                result=super().batched_forward_pass(*a,**kw)
            else:
                from head_runtime import batched_forward
                result=batched_forward(self,*a,**kw)
            if phase=='old_policy':self.raw_masks=result[3].detach()
            if args.strict:
                audit['output_sha256']=[tensor_sha(t) if isinstance(t,torch.Tensor) else None for t in result]
                audit['value_first']=result[2].detach().cpu().flatten()[:4].tolist()
                with (output/'FORWARD_AUDIT.jsonl').open('a') as handle:handle.write(json.dumps(audit)+'\n')
            return result

        def _kl_penalty(self,logp,logref):
            return repair_policy.penalty(self,logp,logref)

        def step(self,*a,**kw):
            phase_started=time.perf_counter()
            stats=super().step(*a,**kw)
            torch.cuda.synchronize()
            self.pending['ppo_step_seconds']=time.perf_counter()-phase_started
            self.pending['trl_phase_seconds']={k:float(v) for k,v in stats.items() if k.startswith('time/')}
            repair_policy.record(self,stats)
            assert self.calls=={'optimizer':config['ppo_epochs'],'backward':config['ppo_epochs']},self.calls
            assert self.forward_calls=={'old_policy':1,'reference':1,'train_forward':config['ppo_epochs']},self.forward_calls
            self.pending.update(loss=float(stats['ppo/loss/total']),kl=float(stats['objective/kl']),
                reward_mean=sum(self.pending['scores'])/3,learning_rate=float(stats['ppo/learning_rate']),
                update_calls=dict(self.calls),forward_calls=dict(self.forward_calls))
            assert all(math.isfinite(self.pending[key]) for key in ['loss','kl','reward_mean','learning_rate'])
            assert list(self.raw.pretrained_model.active_adapters)==['default']
            return stats

        def finish_batch(self):
            if self.pending is None:return
            self.pending['cleanup']=cleanup.finish();torch.cuda.synchronize()
            if hasattr(self,'_m9_last_audit'):self.pending['m9_head']=self._m9_last_audit
            row=self.pending;row['wall_seconds']=time.perf_counter()-row.pop('started')
            row['peak_allocated_bytes']=torch.cuda.max_memory_allocated();row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            memory=torch.cuda.memory_stats();row['allocator_ooms']=memory.get('num_ooms',0);row['allocator_retries']=memory.get('num_alloc_retries',0)
            if hasattr(self,'_compiled_rollout') and hasattr(self._compiled_rollout,'statistics'):
                row['migration_runtime_cumulative']=self._compiled_rollout.statistics()
            self.rows.append(row)
            with (output/'batches.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
            elapsed=self.elapsed_before_resume+time.perf_counter()-started
            all_tokens=sum(r['input_tokens']+r['response_tokens'] for r in self.rows)
            recent=self.rows[-100:]
            write(output/'PROGRESS.json',dict(batch=row['batch'],total_batches=2491,run_stop_batch=args.limit,
                examples=len(self.rollouts),elapsed_seconds=elapsed,input_plus_response_tokens_per_second=all_tokens/elapsed,
                response_tokens_per_second=sum(r['response_tokens'] for r in self.rows)/elapsed,
                recent_seconds_per_batch=sum(r['wall_seconds'] for r in recent)/len(recent),
                recent_math_reward_rate=sum(r['reward_mean'] for r in recent)/len(recent),
                recent_eos_stop_rate=sum(r['eos_stopped'] for r in recent)/(3*len(recent)),
                recent_k3_kl=sum(r['k3_kl'] for r in recent)/len(recent),
                recent_raw_signed_kl=sum(r['raw_signed_kl'] for r in recent)/len(recent),
                quality_zero_reward_streak=self.zero_reward_streak,
                loss=row['loss'],kl=row['kl'],reward_mean=row['reward_mean'],variant=args.variant,
                peak_allocated_gib=max(r['peak_allocated_bytes'] for r in self.rows)/2**30,
                peak_reserved_gib=max(r['peak_reserved_bytes'] for r in self.rows)/2**30,
                allocator_ooms=row['allocator_ooms'],allocator_retries=row['allocator_retries']))
            self.pending=None

    workflow.CustomPPOTrainer=GSM8KPPOTrainer
    def terminate_after_batch(signum,frame):
        if trainer_holder:trainer_holder[0].control.should_training_stop=True
    signal.signal(signal.SIGTERM,terminate_after_batch)
    try:
        workflow.run_ppo(*get_train_args(config),callbacks=[AuditCallback()])
        trainer=trainer_holder[0];trainer.check_frozen_reference()
        assert len(trainer.rows)==trainer.state.global_step
        assert len(trainer.rollouts)==3*trainer.state.global_step
        visited=collections.Counter(row['index'] for row in trainer.rollouts)
        full=trainer.state.global_step==2491 and len(visited)==7473 and set(visited.values())=={1}
        if trainer.quality_stop_reason:
            write(output/'QUALITY_STOPPED.json',dict(reason=trainer.quality_stop_reason,batches=trainer.state.global_step,checkpoint=read(output/'LATEST_CHECKPOINT.json')))
            raise RuntimeError('Quality stop: '+trainer.quality_stop_reason)
        assert trainer.state.global_step==args.limit,'Bounded run stopped early; retain full state and inspect resource ledger'
        assert sha(Path(request['sft_checkpoint'])/'adapter_model.safetensors')==request['sft_adapter_sha256']
        elapsed=trainer.elapsed_before_resume+time.perf_counter()-started
        totals=dict(input_tokens=sum(row['input_tokens'] for row in trainer.rows),response_tokens=sum(row['response_tokens'] for row in trainer.rows))
        result=dict(full_dataset_completed=full,batches=len(trainer.rows),examples=len(trainer.rollouts),
            unique_examples=len(visited),exactly_once=full,elapsed_seconds=elapsed,**totals,
            input_plus_response_tokens_per_second=sum(totals.values())/elapsed,
            response_tokens_per_second=totals['response_tokens']/elapsed,
            reference_unchanged=True,original_sft_checkpoint_unchanged=True,
            all_ppo_updates_per_batch=config['ppo_epochs'],optimizer_updates=len(trainer.rows)*config['ppo_epochs'],variant=args.variant,strict=args.strict,stress=args.stress,
            max_allocated_gib=max(row['peak_allocated_bytes'] for row in trainer.rows)/2**30,
            max_reserved_gib=max(row['peak_reserved_bytes'] for row in trainer.rows)/2**30,
            allocator_ooms=max(row['allocator_ooms'] for row in trainer.rows),allocator_retries=max(row['allocator_retries'] for row in trainer.rows),
            saved_model=str(output/'model'),complete_checkpoint=read(output/'LATEST_CHECKPOINT.json'),
            reward_model=reward['model'],reward_revision=reward['revision'])
        if hasattr(trainer,'_compiled_rollout'):
            result['compile_initialization_seconds']=trainer._compiled_rollout.initialization_seconds
            result['compiled_cache_length']=trainer._compiled_rollout.max_cache_len
            result['compile_counters']={key:dict(value) for key,value in torch._dynamo.utils.counters.items()}
        result['portable_runtime']=options
        if hasattr(trainer,'_compiled_rollout') and hasattr(trainer._compiled_rollout,'statistics'):
            result['migration_runtime']=trainer._compiled_rollout.statistics()
        write(output/'SUCCESS.json',result);print(json.dumps(result,indent=2),flush=True)
    finally:
        cleanup.restore()
        if torch.distributed.is_available() and torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
