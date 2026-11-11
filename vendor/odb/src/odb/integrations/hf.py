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

"""HuggingFace Trainer adapter for ODB."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal

from odb.callbacks import ODBDynamicBatchCallback, _wrap_compute_loss
from odb.core import apply
from odb.handle import ODBHandle
from odb.step_info import ODBStepInfo, pop_step_info

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import Trainer

try:
    from transformers import Trainer as _TransformersTrainer

    _TRANSFORMERS_TRAINER_AVAILABLE = True
except ImportError:
    _TransformersTrainer = object
    _TRANSFORMERS_TRAINER_AVAILABLE = False


MaxStepsPolicy = Literal["error", "overwrite", "preserve"]
SchedulerProgress = Literal["samples", "optimizer_steps"]


@dataclass
class ODBTrainerBridgeState:
    emitted_samples: int = 0
    optimizer_steps: int = 0
    last_step_info: ODBStepInfo | None = None


@dataclass
class ODBTrainerBridge:
    trainer: "Trainer"
    handle: ODBHandle
    sample_budget: int
    max_optimizer_steps: int | None = None
    scheduler_progress: SchedulerProgress = "samples"
    state: ODBTrainerBridgeState = field(default_factory=ODBTrainerBridgeState)

    def consume_step(self, info: ODBStepInfo) -> None:
        self.state.last_step_info = info
        self.state.emitted_samples += int(info.all_samples_this_step)


def _record_step_info(trainer: object, step_info: ODBStepInfo) -> None:
    state = getattr(trainer, "state", None)
    if state is None:
        return
    if hasattr(state, "odb_step_infos"):
        state.odb_step_infos.append(step_info)
    elif hasattr(state, "total_batch_sizes"):
        state.total_batch_sizes.append(step_info.all_samples_this_step)


def _scale_loss_output(trainer: object, loss, step_info: ODBStepInfo):
    scale = step_info.loss_scale
    should_scale = not isinstance(scale, (int, float)) or float(scale) != 1.0
    if not should_scale:
        return loss

    target_loss = loss[0] if isinstance(loss, tuple) else loss
    if hasattr(target_loss, "detach") and hasattr(target_loss, "item"):
        if not hasattr(trainer, "_odb_unscaled_loss_sum"):
            trainer._odb_unscaled_loss_sum = [0.0]
            trainer._odb_unscaled_loss_count = [0]
        trainer._odb_unscaled_loss_sum[0] += target_loss.detach().item()
        trainer._odb_unscaled_loss_count[0] += 1

    scaled_loss = target_loss * scale
    if isinstance(loss, tuple):
        return (scaled_loss, *loss[1:])
    return scaled_loss


class ODBTrainerMixin:
    """Mixin for framework-native Trainer integrations.

    Put this mixin before the concrete Trainer class in the inheritance list:

    ``class MyTrainer(ODBTrainerMixin, Trainer): ...``
    """

    odb_loss_scaling: bool | str | None = "auto"

    def set_odb_loss_scaling(self, loss_scaling: bool | str | None) -> "ODBTrainerMixin":
        self.odb_loss_scaling = loss_scaling
        return self

    def compute_loss(self, model, inputs, *args, **kwargs):
        if not isinstance(inputs, dict):
            return super().compute_loss(model, inputs, *args, **kwargs)

        step_info = pop_step_info(inputs, loss_scaling=self.odb_loss_scaling)
        _record_step_info(self, step_info)
        loss = super().compute_loss(model, inputs, *args, **kwargs)
        return _scale_loss_output(self, loss, step_info)


class ODBTrainer(ODBTrainerMixin, _TransformersTrainer):
    """HuggingFace ``Trainer`` with native ODB metadata consumption."""

    if not _TRANSFORMERS_TRAINER_AVAILABLE:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ODBTrainer requires the `transformers` package. Install with: pip install online-dynamic-batching[hf]"
            )


def _set_trainer_max_steps(
    args,
    *,
    sample_budget: int,
    max_optimizer_steps: int | None,
    policy: MaxStepsPolicy,
) -> None:
    if policy not in {"error", "overwrite", "preserve"}:
        raise ValueError("max_steps_policy must be 'error', 'overwrite', or 'preserve'")

    desired = int(max_optimizer_steps) if max_optimizer_steps is not None else int(sample_budget)
    existing = getattr(args, "max_steps", None)
    existing_active = existing is not None and int(existing) > 0

    if existing_active and int(existing) != desired:
        if policy == "error":
            raise ValueError(
                "trainer.args.max_steps is already set to "
                f"{existing}, but ODB wants {desired}. Pass max_steps_policy='overwrite' "
                "or 'preserve' explicitly."
            )
        if policy == "preserve":
            return

    args.max_steps = desired


def _active_max_steps(args: object | None) -> int | None:
    value = getattr(args, "max_steps", None)
    if value is None:
        return None
    try:
        steps = int(value)
    except (TypeError, ValueError):
        return None
    return steps if steps > 0 else None


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


def _infer_sample_budget(
    trainer: "Trainer",
    *,
    train_dataloader: "DataLoader | None",
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
        if train_dataset is None and train_dataloader is not None:
            train_dataset = getattr(train_dataloader, "dataset", None)
        if train_dataset is None:
            train_dataset = getattr(trainer, "train_dataset", None)
        if train_dataset is not None:
            try:
                dataset_size = len(train_dataset)  # type: ignore[arg-type]
            except TypeError:
                dataset_size = None

    if dataset_size is None:
        raise ValueError(
            "HF Trainer ODB integration could not infer dataset size. "
            "Pass sample_budget=..., dataset_size=..., or train_dataset=...."
        )
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be > 0, got {dataset_size}")

    if num_train_epochs is None:
        num_train_epochs = getattr(getattr(trainer, "args", None), "num_train_epochs", None)
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
            "HF Trainer ODB integration requires per_device_train_batch_size=1. "
            f"Got per_device_train_batch_size={batch_size}."
        )


def _validate_dataloader(
    dataloader: "DataLoader | None",
    *,
    require_batch_size_one: bool,
    require_workers: bool,
) -> None:
    if dataloader is None:
        return

    batch_size = getattr(dataloader, "batch_size", None)
    if require_batch_size_one and batch_size is not None and int(batch_size) != 1:
        raise ValueError(f"HF Trainer ODB integration requires DataLoader batch_size=1. Got batch_size={batch_size}.")

    num_workers = getattr(dataloader, "num_workers", None)
    if require_workers and num_workers is not None and int(num_workers) <= 0:
        raise ValueError(
            "HF Trainer ODB integration requires worker prefetching before grouping. "
            f"Got DataLoader num_workers={num_workers}; set dataloader_num_workers > 0."
        )


def _contains_input_ids(sample: Any) -> bool:
    if isinstance(sample, dict):
        return "input_ids" in sample and sample["input_ids"] is not None
    if isinstance(sample, (list, tuple)):
        return any(_contains_input_ids(item) for item in sample)
    return False


def _declares_odb_ready(dataset: object | None) -> bool:
    if dataset is None:
        return False
    for name in ("odb_ready", "__odb_ready__"):
        if bool(getattr(dataset, name, False)):
            return True
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
    except Exception as exc:  # pragma: no cover - depends on user datasets/processors
        raise ValueError(
            "HF Trainer ODB pipeline readiness check could not read train_dataset[0]. "
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

    if _declares_odb_ready(dataset):
        return

    sample = _sample_from_dataset(dataset)
    if sample is None or _contains_input_ids(sample):
        return

    raise ValueError(
        "HF Trainer ODB requires an ODB-ready lazy tensor sample pipeline: each dataset item "
        "must already contain input_ids before ODB grouping. The sampled item did not contain "
        "input_ids, which usually means tokenizer/processor work is still happening in the collator "
        "after grouping. Move single-sample tensorization into the Dataset or pass validate_pipeline=False "
        "only after auditing that the grouping length is already post-processor length."
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
    dataloader: "DataLoader | None" = None,
    handle: ODBHandle | None = None,
    sample_budget: int,
    max_optimizer_steps: int | None = None,
    scheduler_progress: SchedulerProgress = "samples",
    max_steps_policy: MaxStepsPolicy = "error",
    wrap_compute_loss: bool = True,
    replace_train_dataloader: bool = True,
    disable_dataloader_len: bool = True,
    **apply_kwargs,
) -> ODBTrainerBridge:
    """Configure a HuggingFace Trainer to consume an ODB dataloader.

    ``sample_budget`` is the logical emitted-sample target. ``max_optimizer_steps``
    is an optional optimizer-update cap. HuggingFace ``args.max_steps`` is set
    only according to ``max_steps_policy``.
    """
    if scheduler_progress not in {"samples", "optimizer_steps"}:
        raise ValueError("scheduler_progress must be 'samples' or 'optimizer_steps'")
    if sample_budget <= 0:
        raise ValueError(f"sample_budget must be > 0, got {sample_budget}")

    if handle is None:
        if dataloader is None:
            raise ValueError("configure_trainer requires either handle=... or dataloader=...")
        handle = apply(dataloader, **apply_kwargs)
    elif apply_kwargs:
        raise ValueError("Do not pass ODB apply arguments when handle is already provided.")

    _set_trainer_max_steps(
        trainer.args,
        sample_budget=sample_budget,
        max_optimizer_steps=max_optimizer_steps,
        policy=max_steps_policy,
    )

    if dataloader is not None:
        if disable_dataloader_len:
            type(dataloader).__len__ = lambda self: None
        if replace_train_dataloader:
            trainer.get_train_dataloader = MethodType(lambda self: dataloader, trainer)

    trainer.add_callback(
        ODBDynamicBatchCallback(
            sample_budget=sample_budget,
            scheduler_progress=scheduler_progress,
            max_optimizer_steps=max_optimizer_steps,
        )
    )

    uses_native_trainer = isinstance(trainer, ODBTrainerMixin)
    if uses_native_trainer:
        trainer.set_odb_loss_scaling(handle.config.loss_scaling)

    if wrap_compute_loss and not uses_native_trainer:
        _wrap_compute_loss(trainer, loss_scaling=handle.config.loss_scaling)

    return ODBTrainerBridge(
        trainer=trainer,
        handle=handle,
        sample_budget=sample_budget,
        max_optimizer_steps=max_optimizer_steps,
        scheduler_progress=scheduler_progress,
    )


def enable_odb(
    trainer: "Trainer",
    *,
    train_dataloader: "DataLoader | None" = None,
    dataloader: "DataLoader | None" = None,
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
    require_batch_size_one: bool = True,
    require_workers: bool = True,
    validate_pipeline: bool = True,
    **apply_kwargs,
) -> ODBTrainerBridge:
    """Enable ODB for a HuggingFace Trainer and an ODB-ready DataLoader.

    HF Trainer can train multimodal models once batches already contain model
    tensors. For ODB, the model-specific processor/tokenizer/vision expansion
    must run before grouping, so this high-level entry point validates that
    sampled dataset items already contain ``input_ids``.
    """
    if train_dataloader is not None and dataloader is not None and train_dataloader is not dataloader:
        raise ValueError("Pass only one of train_dataloader=... or dataloader=....")
    dataloader = train_dataloader if dataloader is None else dataloader
    if dataloader is None:
        dataloader = trainer.get_train_dataloader()

    args = getattr(trainer, "args", None)
    _validate_training_args(args, require_batch_size_one=require_batch_size_one)
    _validate_dataloader(
        dataloader,
        require_batch_size_one=require_batch_size_one,
        require_workers=require_workers,
    )
    _validate_pipeline_ready(
        dataloader=dataloader,
        train_dataset=train_dataset,
        validate_pipeline=validate_pipeline,
    )

    resolved_sample_budget = _infer_sample_budget(
        trainer,
        train_dataloader=dataloader,
        train_dataset=train_dataset,
        dataset_size=dataset_size,
        num_train_epochs=num_train_epochs,
        sample_budget=sample_budget,
    )
    if max_optimizer_steps is None:
        max_optimizer_steps = _active_max_steps(args)

    resolved_token_budget = token_budget
    if resolved_token_budget is None:
        resolved_token_budget = _first_attr(args, names=("odb_token_budget", "token_budget", "max_input_length"))
    if resolved_token_budget is None:
        raise ValueError("HF Trainer ODB integration requires token_budget=... or trainer.args.odb_token_budget.")

    _put_apply_arg(apply_kwargs, "token_budget", resolved_token_budget)
    _put_apply_arg(apply_kwargs, "loss_scaling", loss_scaling)
    _put_apply_arg(apply_kwargs, "join", join)

    return configure_trainer(
        trainer,
        dataloader=dataloader,
        sample_budget=resolved_sample_budget,
        max_optimizer_steps=max_optimizer_steps,
        scheduler_progress=scheduler_progress,
        max_steps_policy=max_steps_policy,
        wrap_compute_loss=wrap_compute_loss,
        replace_train_dataloader=replace_train_dataloader,
        disable_dataloader_len=disable_dataloader_len,
        **apply_kwargs,
    )


__all__ = [
    "MaxStepsPolicy",
    "ODBTrainer",
    "ODBTrainerBridge",
    "ODBTrainerBridgeState",
    "ODBTrainerMixin",
    "SchedulerProgress",
    "configure_trainer",
    "enable_odb",
]
