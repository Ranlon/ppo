# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLaMA-Factory adapter for ODB.

LLaMA-Factory is built on HuggingFace Trainer, but its integration point usually
lives in LLaMA-Factory's argument/config plumbing. This module resolves that
surface into the generic HF Trainer adapter without making ODB depend on
LLaMA-Factory internals.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from odb.core import apply as apply_odb
from odb.handle import ODBHandle

from .hf import MaxStepsPolicy, ODBTrainerBridge, SchedulerProgress, configure_trainer as configure_hf_trainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import Trainer


def _first_attr(*objects: object, names: tuple[str, ...]) -> Any | None:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
    return None


def _active_max_steps(args: object | None) -> int | None:
    value = getattr(args, "max_steps", None)
    if value is None:
        return None
    try:
        steps = int(value)
    except (TypeError, ValueError):
        return None
    return steps if steps > 0 else None


def _infer_sample_budget(
    trainer: "Trainer",
    *,
    training_args: object | None,
    train_dataset: object | None,
    dataset_size: int | None,
    num_train_epochs: float | int | None,
    sample_budget: int | None,
) -> int:
    if sample_budget is not None:
        if sample_budget <= 0:
            raise ValueError(f"sample_budget must be > 0, got {sample_budget}")
        return int(sample_budget)

    if dataset_size is None:
        if train_dataset is None:
            train_dataset = getattr(trainer, "train_dataset", None)
        if train_dataset is not None:
            try:
                dataset_size = len(train_dataset)  # type: ignore[arg-type]
            except TypeError:
                dataset_size = None

    if dataset_size is None:
        raise ValueError(
            "LLaMA-Factory ODB configure_trainer could not infer dataset size. "
            "Pass sample_budget=..., dataset_size=..., or train_dataset=...."
        )
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be > 0, got {dataset_size}")

    if num_train_epochs is None:
        num_train_epochs = getattr(training_args, "num_train_epochs", None)
    if num_train_epochs is None:
        num_train_epochs = 1

    budget = math.ceil(int(dataset_size) * float(num_train_epochs))
    if budget <= 0:
        raise ValueError(
            "sample budget inferred from dataset_size and num_train_epochs must be > 0, "
            f"got dataset_size={dataset_size}, num_train_epochs={num_train_epochs}"
        )
    return budget


def _validate_training_args(args: object | None, *, require_batch_size_one: bool) -> None:
    if args is None or not require_batch_size_one:
        return

    batch_size = getattr(args, "per_device_train_batch_size", None)
    if batch_size is not None and int(batch_size) != 1:
        raise ValueError(
            "LLaMA-Factory ODB integration requires per_device_train_batch_size=1. "
            f"Got per_device_train_batch_size={batch_size}."
        )


def _validate_dataloader(dataloader: "DataLoader | None", *, require_batch_size_one: bool) -> None:
    if dataloader is None:
        return

    batch_size = getattr(dataloader, "batch_size", None)
    if require_batch_size_one and batch_size is not None and int(batch_size) != 1:
        raise ValueError(
            f"LLaMA-Factory ODB integration requires DataLoader batch_size=1. Got batch_size={batch_size}."
        )

    num_workers = getattr(dataloader, "num_workers", None)
    if num_workers is not None and int(num_workers) <= 0:
        raise ValueError(
            "LLaMA-Factory ODB integration requires worker prefetching before grouping. "
            f"Got DataLoader num_workers={num_workers}; set dataloader_num_workers > 0."
        )


def _contains_input_ids(sample: Any) -> bool:
    if isinstance(sample, dict):
        return "input_ids" in sample and sample["input_ids"] is not None
    if isinstance(sample, (list, tuple)):
        return any(_contains_input_ids(item) for item in sample)
    return False


def _sample_from_dataset(dataset: object | None) -> Any | None:
    if dataset is None:
        return None
    try:
        size = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        size = None
    if size is not None and size <= 0:
        return None
    try:
        return dataset[0]  # type: ignore[index]
    except Exception as exc:  # pragma: no cover - depends on framework datasets
        raise ValueError(
            "LLaMA-Factory ODB pipeline readiness check could not read train_dataset[0]. "
            "Pass validate_pipeline=False only if you have already verified that each dataset item "
            "is an ODB-ready model tensor sample containing input_ids."
        ) from exc


def _validate_pipeline_ready(
    *,
    dataloader: "DataLoader | None",
    train_dataset: object | None,
    validate_pipeline: bool,
) -> None:
    if not validate_pipeline:
        return

    dataset = getattr(dataloader, "dataset", None) if dataloader is not None else None
    if dataset is None:
        dataset = train_dataset

    sample = _sample_from_dataset(dataset)
    if sample is None:
        return
    if _contains_input_ids(sample):
        return

    raise ValueError(
        "LLaMA-Factory ODB requires an ODB-ready lazy tensor sample pipeline: each dataset item "
        "must already contain input_ids before ODB grouping. The sampled item did not contain "
        "input_ids, which usually means tokenizer/processor/collator work is still happening after "
        "grouping. Keep LLaMA-Factory's dataset/template/mm_plugin semantics, but move the "
        "single-sample tensorization before ODB or use the ODB integration skill to patch the fork."
    )


def _put_apply_arg(kwargs: dict[str, Any], name: str, value: Any | None) -> None:
    if value is None:
        return
    if name in kwargs and kwargs[name] != value:
        raise ValueError(f"Conflicting ODB apply argument {name}: {kwargs[name]!r} vs {value!r}")
    kwargs[name] = value


def configure_trainer(
    trainer: "Trainer",
    *,
    train_dataloader: "DataLoader | None" = None,
    dataloader: "DataLoader | None" = None,
    handle: ODBHandle | None = None,
    training_args: object | None = None,
    data_args: object | None = None,
    finetuning_args: object | None = None,
    train_dataset: object | None = None,
    dataset_size: int | None = None,
    num_train_epochs: float | int | None = None,
    sample_budget: int | None = None,
    token_budget: int | None = None,
    loss_scaling: bool | str | None = None,
    join: bool | None = None,
    max_optimizer_steps: int | None = None,
    scheduler_progress: SchedulerProgress = "samples",
    max_steps_policy: MaxStepsPolicy = "overwrite",
    wrap_compute_loss: bool = True,
    replace_train_dataloader: bool = True,
    disable_dataloader_len: bool = True,
    trainer_integration: Literal["package", "framework"] = "package",
    require_batch_size_one: bool = True,
    **apply_kwargs,
) -> ODBTrainerBridge:
    """Configure a LLaMA-Factory Trainer for ODB.

    The adapter keeps LLaMA-Factory-specific concerns here:

    - infer ``sample_budget`` from ``train_dataset`` / ``dataset_size`` and
      ``num_train_epochs``;
    - keep Trainer stopping/progress settings compatible with ODB's
      sample-budget accounting;
    - check that ``per_device_train_batch_size`` is ``1`` before ODB changes
      batch size dynamically;
    - resolve common ODB config names such as ``odb_token_budget``,
      ``odb_loss_scaling``, and ``odb_join``.
    """
    if train_dataloader is not None and dataloader is not None and train_dataloader is not dataloader:
        raise ValueError("Pass only one of train_dataloader=... or dataloader=....")
    dataloader = train_dataloader if dataloader is None else dataloader
    training_args = training_args if training_args is not None else getattr(trainer, "args", None)

    _validate_training_args(training_args, require_batch_size_one=require_batch_size_one)

    resolved_sample_budget = _infer_sample_budget(
        trainer,
        training_args=training_args,
        train_dataset=train_dataset,
        dataset_size=dataset_size,
        num_train_epochs=num_train_epochs,
        sample_budget=sample_budget,
    )

    if max_optimizer_steps is None:
        max_optimizer_steps = _active_max_steps(training_args)

    if trainer_integration not in {"package", "framework"}:
        raise ValueError("trainer_integration must be 'package' or 'framework'")

    if handle is None:
        resolved_token_budget = token_budget
        if resolved_token_budget is None:
            resolved_token_budget = _first_attr(
                training_args,
                data_args,
                finetuning_args,
                names=("odb_token_budget", "token_budget", "max_input_length"),
            )
        resolved_loss_scaling = loss_scaling
        if resolved_loss_scaling is None:
            resolved_loss_scaling = _first_attr(
                training_args,
                data_args,
                finetuning_args,
                names=("odb_loss_scaling", "loss_scaling"),
            )
        if resolved_loss_scaling is None:
            resolved_loss_scaling = "approx"
        resolved_join = join
        if resolved_join is None:
            resolved_join = _first_attr(training_args, data_args, finetuning_args, names=("odb_join", "join"))

        _put_apply_arg(apply_kwargs, "token_budget", resolved_token_budget)
        _put_apply_arg(apply_kwargs, "loss_scaling", resolved_loss_scaling)
        _put_apply_arg(apply_kwargs, "join", resolved_join)
    elif token_budget is not None or loss_scaling is not None or join is not None or apply_kwargs:
        raise ValueError("Do not pass ODB apply arguments when handle is already provided.")

    if trainer_integration == "framework":
        if handle is None:
            if dataloader is None:
                raise ValueError("framework trainer integration requires either handle=... or dataloader=...")
            handle = apply_odb(dataloader, **apply_kwargs)

        return ODBTrainerBridge(
            trainer=trainer,
            handle=handle,
            sample_budget=resolved_sample_budget,
            max_optimizer_steps=max_optimizer_steps,
            scheduler_progress=scheduler_progress,
        )

    return configure_hf_trainer(
        trainer,
        dataloader=dataloader,
        handle=handle,
        sample_budget=resolved_sample_budget,
        max_optimizer_steps=max_optimizer_steps,
        scheduler_progress=scheduler_progress,
        max_steps_policy=max_steps_policy,
        wrap_compute_loss=wrap_compute_loss,
        replace_train_dataloader=replace_train_dataloader,
        disable_dataloader_len=disable_dataloader_len,
        **apply_kwargs,
    )


def enable_odb(
    trainer: "Trainer",
    *,
    train_dataloader: "DataLoader | None" = None,
    dataloader: "DataLoader | None" = None,
    training_args: object | None = None,
    data_args: object | None = None,
    finetuning_args: object | None = None,
    train_dataset: object | None = None,
    dataset_size: int | None = None,
    num_train_epochs: float | int | None = None,
    sample_budget: int | None = None,
    token_budget: int | None = None,
    loss_scaling: bool | str | None = "exact",
    join: bool | None = True,
    max_optimizer_steps: int | None = None,
    scheduler_progress: SchedulerProgress = "samples",
    max_steps_policy: MaxStepsPolicy = "overwrite",
    wrap_compute_loss: bool = True,
    replace_train_dataloader: bool = True,
    disable_dataloader_len: bool = True,
    trainer_integration: Literal["package", "framework"] = "package",
    require_batch_size_one: bool = True,
    validate_pipeline: bool = True,
    **apply_kwargs,
) -> ODBTrainerBridge:
    """Enable ODB for a LLaMA-Factory-style Trainer and DataLoader.

    This is the high-level integration entry point. It assumes LLaMA-Factory's
    own dataset/template/mm_plugin path has already produced ODB-ready
    single-sample tensor dicts before grouping. ODB does not reimplement
    LLaMA-Factory processors.
    """
    if train_dataloader is not None and dataloader is not None and train_dataloader is not dataloader:
        raise ValueError("Pass only one of train_dataloader=... or dataloader=....")
    dataloader = train_dataloader if dataloader is None else dataloader

    _validate_training_args(
        training_args if training_args is not None else getattr(trainer, "args", None),
        require_batch_size_one=require_batch_size_one,
    )
    _validate_dataloader(dataloader, require_batch_size_one=require_batch_size_one)
    _validate_pipeline_ready(dataloader=dataloader, train_dataset=train_dataset, validate_pipeline=validate_pipeline)

    return configure_trainer(
        trainer,
        dataloader=dataloader,
        training_args=training_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
        train_dataset=train_dataset,
        dataset_size=dataset_size,
        num_train_epochs=num_train_epochs,
        sample_budget=sample_budget,
        token_budget=token_budget,
        loss_scaling=loss_scaling,
        join=join,
        max_optimizer_steps=max_optimizer_steps,
        scheduler_progress=scheduler_progress,
        max_steps_policy=max_steps_policy,
        wrap_compute_loss=wrap_compute_loss,
        replace_train_dataloader=replace_train_dataloader,
        disable_dataloader_len=disable_dataloader_len,
        trainer_integration=trainer_integration,
        require_batch_size_one=require_batch_size_one,
        **apply_kwargs,
    )


__all__ = ["configure_trainer", "enable_odb"]
