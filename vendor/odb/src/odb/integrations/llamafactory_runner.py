"""Run the same LLaMA-Factory SFT config with or without ODB.

Launch under torchrun so every rank installs the hook before LLaMA-Factory
constructs its Trainer. The LLaMA-Factory checkout is never modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def install_odb_hook(
    trainer_cls: type,
    *,
    sample_budget: int | None,
    token_budget: int,
    buffer_size: int | None = None,
    full_epoch: bool = False,
) -> None:
    """Attach ODB to the prepared SFT training loader once per Trainer."""
    if (sample_budget is not None and sample_budget <= 0) or token_budget <= 0:
        raise ValueError("sample_budget and token_budget must be positive")
    if full_epoch and sample_budget is not None:
        raise ValueError("full_epoch derives sample_budget from the prepared training dataset")
    if not full_epoch and sample_budget is None:
        raise ValueError("sample_budget is required outside full_epoch mode")
    if buffer_size is not None and buffer_size <= 0:
        raise ValueError("buffer_size must be positive")

    original_get_train_dataloader = trainer_cls.get_train_dataloader
    original_train = trainer_cls.train

    def get_train_dataloader(self, *args: Any, **kwargs: Any):
        if hasattr(self, "_odb_train_loader"):
            return self._odb_train_loader

        loader = original_get_train_dataloader(self, *args, **kwargs)
        from torch.utils.data import DataLoader

        if type(loader).__name__ == "DataLoaderDispatcher":
            raise ValueError("ODB requires rank-local DataLoaderShard; disable Accelerate dispatch_batches")
        # Accelerate 1.x wraps its DataLoader in DataLoaderShard. Its iterator
        # reads base_dataloader, so applying ODB to the outer object is inert.
        odb_loader = getattr(loader, "base_dataloader", loader)
        if not isinstance(odb_loader, DataLoader):
            raise TypeError(f"ODB requires a PyTorch DataLoader; got {type(odb_loader).__name__}")
        # Accelerate commonly reconstructs DataLoaderShard with a BatchSamplerShard,
        # which makes PyTorch report batch_size=None even when every index batch is 1.
        batch_size = odb_loader.batch_size
        if batch_size is None:
            batch_size = getattr(odb_loader.batch_sampler, "batch_size", None)
            if batch_size == 1:
                object.__setattr__(odb_loader, "batch_size", 1)
        if batch_size != 1:
            raise ValueError(f"ODB requires prepared DataLoader batch_size=1; got {batch_size!r}")
        if odb_loader.num_workers < 1:
            raise ValueError("ODB requires dataloader_num_workers > 0")
        if odb_loader.persistent_workers:
            raise ValueError("ODB does not support dataloader_persistent_workers=true")

        from .llamafactory import enable_odb

        if full_epoch and float(getattr(self.args, "num_train_epochs", 1)) != 1.0:
            raise ValueError("ODB full_epoch requires num_train_epochs=1")
        resolved_sample_budget = len(self.train_dataset) if full_epoch else sample_budget
        if resolved_sample_budget is None or resolved_sample_budget <= 0:
            raise ValueError("ODB requires a nonempty prepared training dataset")
        if full_epoch and self.is_world_process_zero():
            print(f"[ODB] full epoch sample budget: {resolved_sample_budget}", flush=True)
        self._odb_sample_budget = resolved_sample_budget
        self._odb_bridge = enable_odb(
            self,
            train_dataloader=odb_loader,
            train_dataset=self.train_dataset,
            sample_budget=resolved_sample_budget,
            token_budget=token_budget,
            loss_scaling="exact",
            join=True,
            max_optimizer_steps=self.args.max_steps if not full_epoch and self.args.max_steps > 0 else None,
            scheduler_progress="samples" if full_epoch else "optimizer_steps",
            max_steps_policy="preserve",
            replace_train_dataloader=False,
            disable_dataloader_len=False,
            buffer_size=buffer_size,
        )
        # Independently synchronize the actual local batch sizes. This keeps
        # sample-budget stopping and exact token loss scaling correct even if
        # the DataLoader subprocess reports rank-local transport metadata.
        from odb.constants import LOCAL_BATCH_SIZE_KEY, LOCAL_TOKENS_KEY, TOTAL_BATCH_SIZE_KEY, TOTAL_TOKENS_KEY

        original_compute_loss = self.compute_loss
        self._odb_local_seen = 0
        self._odb_local_microbatches = 0
        self._odb_metadata_mismatch_reported = False
        self._odb_metadata_verified = None

        def compute_loss(model, inputs, *loss_args, **loss_kwargs):
            from torch import distributed as dist

            local_samples = int(inputs.get(LOCAL_BATCH_SIZE_KEY, inputs["input_ids"].shape[0]))
            local_tokens = int(inputs[LOCAL_TOKENS_KEY])
            total_samples, total_tokens = local_samples, local_tokens
            reported_total = inputs.get(TOTAL_BATCH_SIZE_KEY)
            reported_tokens = inputs.get(TOTAL_TOKENS_KEY)
            distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
            if distributed and self._odb_metadata_verified is True:
                total_samples, total_tokens = int(reported_total), int(reported_tokens)
            elif distributed:
                import torch

                counts = torch.tensor([local_samples, local_tokens], dtype=torch.long, device=self.args.device)
                dist.all_reduce(counts)
                total_samples, total_tokens = (int(value) for value in counts.tolist())
                if self._odb_metadata_verified is None:
                    self._odb_metadata_verified = (
                        reported_total is not None
                        and reported_tokens is not None
                        and int(reported_total) == total_samples
                        and int(reported_tokens) == total_tokens
                    )

            if reported_total is not None and int(reported_total) != total_samples:
                if not self._odb_metadata_mismatch_reported and self.is_world_process_zero():
                    print(
                        f"[ODB] correcting reported global batch size {int(reported_total)} "
                        f"to {total_samples}; check DataLoader subprocess DDP metadata",
                        flush=True,
                    )
                self._odb_metadata_mismatch_reported = True
            inputs[TOTAL_BATCH_SIZE_KEY] = total_samples
            inputs[TOTAL_TOKENS_KEY] = total_tokens
            loss = original_compute_loss(model, inputs, *loss_args, **loss_kwargs)
            self._odb_local_seen += local_samples
            self._odb_local_microbatches += 1
            return loss

        self.compute_loss = compute_loss
        self._odb_train_loader = loader
        return loader

    def train(self, *args: Any, **kwargs: Any):
        result = original_train(self, *args, **kwargs)
        actual = int(getattr(self, "_odb_local_seen", 0))
        microbatches = int(getattr(self, "_odb_local_microbatches", 0))
        if hasattr(self, "_odb_train_loader"):
            from torch import distributed as dist

            from odb.core import close_odb_loader

            if dist.is_available() and dist.is_initialized():
                import torch

                counts = torch.tensor([actual, microbatches], dtype=torch.long, device=self.args.device)
                dist.all_reduce(counts)
                actual, microbatches = (int(value) for value in counts.tolist())
                dist.barrier()
            odb_loader = getattr(self._odb_train_loader, "base_dataloader", self._odb_train_loader)
            close_odb_loader(odb_loader)
        if actual <= 0:
            raise RuntimeError("ODB finished without accounting for any samples")
        runtime = float(result.metrics.get("train_runtime", 0))
        result.metrics["odb_callback_samples"] = int(getattr(self.state, "total_data_step", 0))
        result.metrics["odb_actual_samples"] = actual
        result.metrics["odb_mean_per_device_batch_size"] = actual / microbatches if microbatches else 0.0
        actual_rate = actual / runtime if runtime > 0 else 0.0
        actual_steps = int(getattr(self.state, "global_step", 0))
        result.metrics["odb_actual_samples_per_second"] = actual_rate
        result.metrics["odb_actual_optimizer_steps"] = actual_steps
        result.metrics["odb_sample_budget"] = int(self._odb_sample_budget)
        result.metrics["odb_token_budget"] = token_budget
        result.metrics["train_samples_per_second"] = actual_rate
        result.metrics["train_steps_per_second"] = actual_steps / runtime if runtime > 0 else 0.0
        return result

    trainer_cls.get_train_dataloader = get_train_dataloader
    trainer_cls.train = train


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "odb"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--llamafactory-source", choices=("installed", "checkout"), default="installed")
    parser.add_argument("--llamafactory-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-budget", type=int, help="Global sample target (default: 800 outside full-epoch mode)")
    parser.add_argument("--full-epoch", action="store_true", help="Train one complete prepared dataset epoch")
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--prefetch-factor", type=int, default=16)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--fixed-batch-size", type=int, help="Per-device batch size for baseline mode")
    parser.add_argument("--baseline-max-steps", type=int, help="Optimizer-step cap for baseline mode")
    return parser.parse_args(argv)


def _apply_baseline_overrides(
    config: dict[str, Any], fixed_batch_size: int | None, max_steps: int | None, *, full_epoch: bool = False
) -> None:
    if fixed_batch_size is not None:
        if fixed_batch_size < 1:
            raise ValueError("fixed-batch-size must be positive")
        if max_steps is None and not full_epoch:
            raise ValueError("fixed-batch-size comparison also requires baseline-max-steps")
        config["per_device_train_batch_size"] = fixed_batch_size
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("baseline-max-steps must be positive")
        config["max_steps"] = max_steps


def _validate_full_epoch_config(config: dict[str, Any], sample_budget: int | None) -> None:
    if sample_budget is not None:
        raise ValueError("--full-epoch cannot be combined with --sample-budget")
    if config.get("max_samples") is not None:
        raise ValueError("--full-epoch requires max_samples to be omitted or null")
    if int(config.get("max_steps", -1)) != -1:
        raise ValueError("--full-epoch requires max_steps: -1")
    if float(config.get("num_train_epochs", 1)) != 1.0:
        raise ValueError("--full-epoch requires num_train_epochs: 1")


def main(argv: list[str] | None = None) -> None:
    options = _parse_args(argv)
    root = options.llamafactory_root.resolve()
    if not options.config.is_file():
        raise FileNotFoundError(options.config)

    source = None
    if options.llamafactory_source == "checkout":
        source = (root / "src").resolve()
        if not (source / "llamafactory" / "train" / "sft" / "trainer.py").is_file():
            raise FileNotFoundError(f"LLaMA-Factory source checkout not found under {root}")
        sys.path.insert(0, str(source))

    import llamafactory

    imported = Path(llamafactory.__file__).resolve()
    if source is not None and source not in imported.parents:
        raise RuntimeError(f"Loaded LLaMA-Factory from {imported}, expected {source}")
    if os.environ.get("RANK", "0") == "0":
        print(f"LLaMA-Factory source: {imported}; mode: {options.mode}", flush=True)

    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(options.config), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("LLaMA-Factory config must be a YAML mapping")
    output_dir = Path(options.output_dir)
    config["output_dir"] = str(output_dir if output_dir.is_absolute() else root / output_dir)
    if config.get("stage") != "sft" or not config.get("do_train"):
        raise ValueError("This runner supports training stage=sft only")
    if options.full_epoch:
        _validate_full_epoch_config(config, options.sample_budget)
    if options.mode == "baseline":
        _apply_baseline_overrides(config, options.fixed_batch_size, options.baseline_max_steps, full_epoch=options.full_epoch)
    else:
        if options.fixed_batch_size is not None or options.baseline_max_steps is not None:
            raise ValueError("fixed-batch-size and baseline-max-steps apply only to baseline mode")
        if int(config.get("per_device_train_batch_size", 1)) != 1:
            raise ValueError("ODB config must use per_device_train_batch_size=1")
        if options.prefetch_factor < 1:
            raise ValueError("prefetch-factor must be positive")
        if not options.full_epoch and int(config.get("max_steps", -1)) <= 0:
            raise ValueError("ODB comparison requires a positive max_steps")
        config["dataloader_prefetch_factor"] = options.prefetch_factor
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

        install_odb_hook(
            CustomSeq2SeqTrainer,
            sample_budget=None if options.full_epoch else (
                options.sample_budget if options.sample_budget is not None else 800
            ),
            token_budget=options.token_budget,
            buffer_size=options.buffer_size,
            full_epoch=options.full_epoch,
        )

    from llamafactory.train.tuner import run_exp

    run_exp(args=config)


if __name__ == "__main__":
    main()
