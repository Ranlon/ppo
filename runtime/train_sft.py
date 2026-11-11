"""Native E0 SFT settings, fresh GSM8K LoRA, observable HF state checkpoints."""
import os
os.environ.update(DISABLE_VERSION_CHECK='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true')
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import sys
import time
from project import run_root,VENDOR
ROOT=run_root()
sys.path.insert(0,str(VENDOR))


def main():
    import torch
    from transformers import TrainerCallback, set_seed
    from adapter_io import install
    install()
    from llamafactory.hparams import get_train_args
    from llamafactory.train.sft import workflow
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=935);parser.add_argument('--name',default='sft');parser.add_argument('--resume-hf')
    args = parser.parse_args()
    assert 0<args.limit<=935 and args.name.replace('_','').isalnum()
    name=args.name
    args.smoke=args.limit<935
    from project import read,write as project_write
    profile=read(ROOT/'PROFILE.json')
    odb_settings=profile['sft'].get('odb')
    if odb_settings and args.resume_hf:
        raise ValueError('ODB SFT resume is not implemented; use a fresh run directory')
    folder = ROOT / name; folder.mkdir(exist_ok=False)
    config = json.loads((ROOT / 'SFT_CONFIG.json').read_text())
    config['output_dir'] = str(folder / 'model')
    if args.smoke:config.update(max_steps=args.limit,logging_steps=1,plot_loss=False)
    if odb_settings:
        config.update(gradient_accumulation_steps=odb_settings['gradient_accumulation_steps'],
            max_steps=args.limit if args.smoke else 7473,dataloader_prefetch_factor=16)
    if args.resume_hf:config['resume_from_checkpoint']=str(Path(args.resume_hf).resolve())
    (folder / 'RUN_CONFIG.json').write_text(json.dumps(config, indent=2) + '\n')
    expected = json.loads((ROOT / 'DATA_PREFLIGHT.json').read_text())
    assert expected['passed'] and 'adapter_name_or_path' not in config
    set_seed(42); started = time.monotonic(); holder = []; seen = collections.Counter()
    native = workflow.CustomSeq2SeqTrainer

    def write(name, value):
        path = folder / name; temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(temp, path)

    def fingerprint(ids, labels):
        return hashlib.sha256(json.dumps([ids, labels], separators=(',', ':')).encode()).hexdigest()

    class AuditedSFT(native):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            assert len(self.train_dataset) == 7473
            expected_accumulation=odb_settings['gradient_accumulation_steps'] if odb_settings else 8
            assert self.args.gradient_accumulation_steps == expected_accumulation and self.args.per_device_train_batch_size == 1
            assert self.args.bf16 and not self.args.fp16 and not self.args.dataloader_drop_last
            assert self.args.num_train_epochs == 1
            if not odb_settings:assert self.args.get_warmup_steps(935) == 29
            params = [p for p in self.model.parameters() if p.requires_grad]
            assert sum(p.numel() for p in params) == 17751040
            assert all(p.dtype == torch.float32 for p in params)
            self.expected = collections.Counter(fingerprint(r['input_ids'], r['labels']) for r in self.train_dataset)
            self.expected_tokens=sum(len(r['input_ids']) for r in self.train_dataset)
            self.expected_supervised=sum(sum(v!=-100 for v in r['labels'][1:]) for r in self.train_dataset)
            self.tokens=0;self.supervised=0;self.examples=0
            self.train_started = None
            assert hasattr(self.model.get_output_embeddings(),'lora_A')
            holder.append(self)

        def create_scheduler(self,num_training_steps,optimizer=None):
            return super().create_scheduler(935,optimizer)

        def training_step(self, model, inputs, *a, **kw):
            for layer in self.model.get_base_model().model.layers:
                layer.gradient_checkpointing=profile['sft']['checkpoint_policy']=='all'
            ids = inputs['input_ids'].detach().cpu().tolist()
            labels = inputs['labels'].detach().cpu().tolist()
            masks = inputs['attention_mask'].detach().cpu().tolist()
            for x, y, mask in zip(ids, labels, masks):
                actual_ids = [v for v, m in zip(x, mask) if m]
                actual_labels = [v for v, m in zip(y, mask) if m]
                seen[fingerprint(actual_ids, actual_labels)] += 1
                self.tokens += len(actual_ids)
                self.supervised += sum(v != -100 for v in actual_labels[1:])
                self.examples += 1
            return super().training_step(model, inputs, *a, **kw)

    if odb_settings:
        from odb_sft import ODBSFTMixin
        class ODBAuditedSFT(ODBSFTMixin, AuditedSFT):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                self.setup_odb(odb_settings)
        workflow.CustomSeq2SeqTrainer = ODBAuditedSFT
    else:
        workflow.CustomSeq2SeqTrainer = AuditedSFT

    class Audit(TrainerCallback):
        def on_train_begin(self, training_args, state, control, **kw):
            holder[0].train_started = time.monotonic(); torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, training_args, state, control, **kw):
            trainer = holder[0]; elapsed = time.monotonic() - trainer.train_started
            if state.global_step == 1: control.should_save = True
            if state.global_step==args.limit:control.should_save=True
            memory = torch.cuda.memory_stats()
            write('PROGRESS.json', dict(stage='sft', updates=state.global_step, total_updates=state.max_steps,
                examples=trainer.examples, total_examples=7473, elapsed_seconds=elapsed,
                input_tokens=trainer.tokens, supervised_tokens=trainer.supervised,
                input_tokens_per_second=trainer.tokens / elapsed, supervised_tokens_per_second=trainer.supervised / elapsed,
                estimated_remaining_seconds=(elapsed / max(1,trainer.examples) * max(0,7473-trainer.examples)
                    if odb_settings else elapsed / state.global_step * (state.max_steps-state.global_step)),
                odb=odb_settings,
                allocator_ooms=memory.get('num_ooms', 0), allocator_retries=memory.get('num_alloc_retries', 0),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
            return control

        def on_log(self, training_args, state, control, logs=None, **kw):
            if logs:
                assert all(math.isfinite(float(v)) for v in logs.values() if isinstance(v, (int, float)))
            return control

    workflow.run_sft(*get_train_args(config), callbacks=[Audit()])
    trainer = holder[0]
    if not args.smoke and not args.resume_hf:
        assert seen == trainer.expected and trainer.examples == 7473
        if not odb_settings:assert trainer.state.global_step == 935
        assert trainer.tokens==trainer.expected_tokens and trainer.supervised==trainer.expected_supervised
    adapter = folder / 'model/adapter_model.safetensors'
    from safetensors.torch import load_file
    tensors = load_file(str(adapter))
    assert len(tensors) == 506 and all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values())
    assert any(torch.count_nonzero(t) for k, t in tensors.items() if 'lora_B' in k)
    checkpoint_dirs = sorted((folder / 'model').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    for path in checkpoint_dirs:
        assert all((path / filename).exists() for filename in ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'])
    memory = torch.cuda.memory_stats(); assert memory.get('num_ooms', 0) == 0
    result = dict(passed=True, smoke=args.smoke, full_dataset_completed=not args.smoke,
        examples=trainer.examples, optimizer_updates=trainer.state.global_step, input_tokens=trainer.tokens,
        supervised_tokens=trainer.supervised, exactly_once=seen==trainer.expected,resume_source=args.resume_hf,checkpoint_policy=profile['sft']['checkpoint_policy'],
        program_seconds=time.monotonic()-started, train_runtime=trainer.state.log_history[-1].get('train_runtime'),
        adapter=str(adapter.parent), adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
        allocator_ooms=memory.get('num_ooms', 0), allocator_retries=memory.get('num_alloc_retries', 0),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        complete_hf_checkpoints=[str(p) for p in checkpoint_dirs], fresh_from_base=True,
        original_alpaca_adapter_not_loaded=True, native_e0_settings_with_output_head_lora=not bool(odb_settings))
    if odb_settings:
        result.update(odb=odb_settings,batch_size_histogram=dict(trainer.odb_batch_sizes),
            mean_batch_size=trainer.examples / sum(trainer.odb_batch_sizes.values()),
            padding_fraction=1-trainer.tokens / trainer.odb_padded_tokens,
            scheduler_progress='consumed_samples',loss_normalization='shifted_supervised_tokens_per_accumulation_window')
    write('SUCCESS.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
