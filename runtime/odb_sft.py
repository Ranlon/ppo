"""Single-device ODB for the audited SFT stage; PPO keeps its legacy TRL stack."""

import math
from collections import Counter

import torch
from transformers import TrainerCallback


def sample_lr_factor(consumed, total, warmup_ratio):
    warmup = max(1, math.ceil(total * warmup_ratio))
    if consumed < warmup:
        return consumed / warmup
    progress = min(1.0, max(0.0, (consumed - warmup) / max(1, total - warmup)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class SampleProgress(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        state.epoch = trainer.examples / len(trainer.train_dataset)
        if trainer.examples >= len(trainer.train_dataset):
            control.should_training_stop = True
            control.should_save = True
        return control


class ODBSFTMixin:
    """Mix into the existing SFT audit class, after its counters are initialized."""

    def setup_odb(self, settings):
        if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
            raise ValueError('PPO ODB SFT currently supports exactly one device')
        if self.args.dataloader_num_workers < 1 or self.args.dataloader_persistent_workers:
            raise ValueError('ODB needs worker processes and persistent_workers=false')
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError('ODB SFT currently requires gradient_accumulation_steps=1 for exact tail handling')
        if self.label_smoother is not None or self.compute_loss_func is not None:
            raise ValueError('ODB supervised-token normalization requires the native causal LM loss')
        self.odb_settings = settings
        self.odb_batch_sizes = Counter()
        self.odb_padded_tokens = 0
        self.odb_loader = None
        # Qwen3 accepts num_items_in_batch. Its CE sum is divided by the total
        # shifted supervised tokens of the whole accumulation window.
        self.model_accepts_loss_kwargs = True
        self.add_callback(SampleProgress(self))

    def get_train_dataloader(self):
        if self.odb_loader is not None:
            return self.odb_loader
        import odb
        loader = super().get_train_dataloader()
        base = getattr(loader, 'base_dataloader', loader)
        if type(loader).__name__ == 'DataLoaderDispatcher':
            raise ValueError('ODB requires rank-local data loading')
        if base.batch_size is None and getattr(base.batch_sampler, 'batch_size', None) == 1:
            object.__setattr__(base, 'batch_size', 1)
        if base.batch_size != 1:
            raise ValueError('ODB input loader must use batch_size=1')
        odb.apply(base, token_budget=self.odb_settings['token_budget'],
                  buffer_size=self.odb_settings['buffer_size'], loss_scaling='none',
                  join=False, no_sync=True, no_warmup=True)
        self.odb_base_loader = base
        self.odb_loader = loader
        return loader

    def _get_num_items_in_batch(self, batch_samples, device):
        if not batch_samples:
            return None
        from odb import pop_step_info
        for batch in batch_samples:
            pop_step_info(batch, loss_scaling='none')
        count = sum((batch['labels'][..., 1:] != -100).sum() for batch in batch_samples)
        if int(count) <= 0:
            raise ValueError('SFT accumulation window contains no supervised tokens')
        return count.to(device)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer or self.optimizer,
                lambda _: sample_lr_factor(self.examples, len(self.train_dataset), self.args.warmup_ratio))
        return self.lr_scheduler

    def training_step(self, model, inputs, *args, **kwargs):
        self.odb_batch_sizes[int(inputs['input_ids'].shape[0])] += 1
        self.odb_padded_tokens += inputs['input_ids'].numel()
        return super().training_step(model, inputs, *args, **kwargs)

    def train(self, *args, **kwargs):
        try:
            result = super().train(*args, **kwargs)
            result.metrics['train_samples_per_second'] = self.examples / result.metrics['train_runtime']
            return result
        finally:
            if getattr(self, 'odb_loader', None) is not None:
                from odb.core import close_odb_loader
                close_odb_loader(self.odb_base_loader)
