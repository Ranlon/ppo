"""CPU integration checks for dynamic SFT coverage and supervised-token loss."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'runtime'))
import project
from odb_sft import ODBSFTMixin, sample_lr_factor


class Samples(Dataset):
    def __len__(self):
        return 17

    def __getitem__(self, index):
        ids = [3 + index] + [22] * (3 + index % 8)
        return dict(input_ids=ids, labels=[-100] + ids[1:])


def collate(samples):
    length = max(len(s['input_ids']) for s in samples)
    return {
        'input_ids': torch.tensor([s['input_ids'] + [0] * (length - len(s['input_ids'])) for s in samples]),
        'labels': torch.tensor([s['labels'] + [-100] * (length - len(s['labels'])) for s in samples]),
        'attention_mask': torch.tensor([[1] * len(s['input_ids']) + [0] * (length - len(s['input_ids'])) for s in samples])}


def model():
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, attention_dropout=0.0))


class AuditedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.examples = 0
        self.seen = []

    def training_step(self, model, inputs, *args, **kwargs):
        self.examples += len(inputs['input_ids'])
        self.seen.extend(inputs['input_ids'][:, 0].tolist())
        return super().training_step(model, inputs, *args, **kwargs)


class DynamicTrainer(ODBSFTMixin, AuditedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_odb(dict(token_budget=24, buffer_size=8))


class ODBSFTTests(unittest.TestCase):
    def test_full_epoch_covers_every_sample_once(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = DynamicTrainer(model=model(), args=TrainingArguments(
                output_dir=folder, use_cpu=True, max_steps=17, gradient_accumulation_steps=1,
                per_device_train_batch_size=1, dataloader_num_workers=1,
                dataloader_prefetch_factor=8, dataloader_pin_memory=False,
                report_to=[], save_strategy='no', logging_strategy='no', disable_tqdm=True),
                train_dataset=Samples(), data_collator=collate)
            trainer.train()
            self.assertEqual(trainer.examples, 17, (trainer.seen, trainer.odb_batch_sizes))
            self.assertEqual(sorted(trainer.seen), list(range(3, 20)))
            self.assertTrue(any(size > 1 for size in trainer.odb_batch_sizes))
            self.assertAlmostEqual(trainer.state.epoch, 1.0)
            self.assertEqual(trainer.lr_scheduler.get_last_lr(), [0.0, 0.0])

    def test_loss_gradient_equals_concatenated_supervised_token_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            original = model()
            reference = copy.deepcopy(original)
            trainer = DynamicTrainer(model=original, args=TrainingArguments(
                output_dir=folder, use_cpu=True, dataloader_num_workers=1, report_to=[]))
            batches = [collate([Samples()[0]]), collate([Samples()[1], Samples()[2]])]
            denominator = trainer._get_num_items_in_batch(batches, torch.device('cpu'))
            for batch in batches:
                trainer.compute_loss(original, batch, num_items_in_batch=denominator).backward()
            reference(**collate([Samples()[i] for i in range(3)])).loss.backward()
            for parameter, expected in zip(original.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, expected.grad, atol=1e-6, rtol=1e-4)

    def test_sample_schedule_boundaries(self):
        self.assertEqual(sample_lr_factor(0, 100, .1), 0)
        self.assertEqual(sample_lr_factor(10, 100, .1), 1)
        self.assertEqual(sample_lr_factor(100, 100, .1), 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
