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

"""PyTorch Lightning adapter for ODB training steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any

from odb.core import apply
from odb.handle import ODBHandle
from odb.step_info import ODBStepInfo, pop_step_info

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

try:
    from lightning.pytorch import Callback as _LightningCallback

    _LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        from pytorch_lightning import Callback as _LightningCallback

        _LIGHTNING_AVAILABLE = True
    except ImportError:
        _LIGHTNING_AVAILABLE = False

        class _LightningCallback:  # type: ignore[no-redef]
            """Stub base class when Lightning is not installed."""


@dataclass
class ODBLightningBridgeState:
    emitted_samples: int = 0
    optimizer_steps: int = 0
    last_step_info: ODBStepInfo | None = None


@dataclass
class ODBLightningBridge:
    """Runtime bridge shared by Lightning mixin and wrapper integrations."""

    module: Any
    handle: ODBHandle
    sample_budget: int | None = None
    loss_scaling: bool | str | None = None
    state: ODBLightningBridgeState = field(default_factory=ODBLightningBridgeState)

    def __post_init__(self) -> None:
        if self.sample_budget is not None and self.sample_budget <= 0:
            raise ValueError(f"sample_budget must be > 0, got {self.sample_budget}")
        if self.loss_scaling is None:
            self.loss_scaling = self.handle.config.loss_scaling

    def consume_batch(self, batch: dict[str, Any]) -> ODBStepInfo:
        info = pop_step_info(batch, loss_scaling=self.loss_scaling)
        self.state.last_step_info = info
        self.state.emitted_samples += int(info.all_samples_this_step)
        setattr(self.module, "odb_last_step_info", info)
        setattr(self.module, "odb_emitted_samples", self.state.emitted_samples)
        return info

    def scale_loss(self, loss, info: ODBStepInfo | None = None):
        step_info = info if info is not None else self.state.last_step_info
        if step_info is None:
            raise RuntimeError("No ODB step info is available. Call consume_batch(batch) before scaling loss.")
        return loss * step_info.loss_scale

    def mark_optimizer_step(self) -> None:
        self.state.optimizer_steps += 1

    @property
    def should_stop(self) -> bool:
        return self.sample_budget is not None and self.state.emitted_samples >= self.sample_budget


class ODBLightningMixin:
    """Mixin for LightningModules that want explicit ODB batch consumption."""

    odb_loss_scaling: bool | str | None = "auto"
    odb_last_step_info: ODBStepInfo | None = None
    odb_emitted_samples: int = 0

    def set_odb_handle(self, handle: ODBHandle, *, loss_scaling: bool | str | None = None) -> "ODBLightningMixin":
        self.odb_handle = handle
        self.odb_loss_scaling = handle.config.loss_scaling if loss_scaling is None else loss_scaling
        return self

    def consume_odb_batch(self, batch: dict[str, Any]) -> ODBStepInfo:
        info = pop_step_info(batch, loss_scaling=self.odb_loss_scaling)
        self.odb_last_step_info = info
        self.odb_emitted_samples += int(info.all_samples_this_step)
        return info

    def scale_odb_loss(self, loss, info: ODBStepInfo | None = None):
        step_info = info if info is not None else self.odb_last_step_info
        if step_info is None:
            raise RuntimeError("No ODB step info is available. Call consume_odb_batch(batch) before scaling loss.")
        return loss * step_info.loss_scale


class ODBLightningCallback(_LightningCallback):
    """Stop Lightning training once an emitted-sample budget is reached."""

    def __init__(self, sample_budget: int):
        if sample_budget <= 0:
            raise ValueError(f"sample_budget must be > 0, got {sample_budget}")
        self.sample_budget = int(sample_budget)

    def on_train_start(self, trainer, pl_module) -> None:
        setattr(pl_module, "odb_emitted_samples", int(getattr(pl_module, "odb_emitted_samples", 0)))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if int(getattr(pl_module, "odb_emitted_samples", 0)) >= self.sample_budget:
            setattr(trainer, "should_stop", True)


def _scale_training_step_output(output, bridge: ODBLightningBridge, info: ODBStepInfo):
    if output is None:
        return None
    if isinstance(output, dict):
        if "loss" not in output:
            return output
        scaled = dict(output)
        scaled["loss"] = bridge.scale_loss(output["loss"], info=info)
        return scaled
    return bridge.scale_loss(output, info=info)


def _wrap_training_step(module: Any, bridge: ODBLightningBridge) -> None:
    if getattr(module, "_odb_training_step_wrapped", False):
        return
    original_training_step = module.training_step

    def _odb_training_step(self, batch, batch_idx, *args, **kwargs):
        info = bridge.consume_batch(batch)
        output = original_training_step(batch, batch_idx, *args, **kwargs)
        return _scale_training_step_output(output, bridge, info)

    module.training_step = MethodType(_odb_training_step, module)
    module._odb_training_step_wrapped = True


def configure_lightning_module(
    module: Any,
    *,
    dataloader: "DataLoader | None" = None,
    handle: ODBHandle | None = None,
    sample_budget: int | None = None,
    wrap_training_step: bool = True,
    loss_scaling: bool | str | None = None,
    **apply_kwargs,
) -> ODBLightningBridge:
    """Configure a LightningModule to consume ODB batches.

    If ``wrap_training_step`` is true, the wrapper pops ODB metadata before the
    module sees the batch and scales tensor or ``{"loss": ...}`` returns. For
    complex modules, use :class:`ODBLightningMixin` and call
    ``consume_odb_batch`` / ``scale_odb_loss`` explicitly.
    """
    if handle is None:
        if dataloader is None:
            raise ValueError("configure_lightning_module requires either handle=... or dataloader=...")
        handle = apply(dataloader, **apply_kwargs)
    elif apply_kwargs:
        raise ValueError("Do not pass ODB apply arguments when handle is already provided.")

    if isinstance(module, ODBLightningMixin):
        module.set_odb_handle(handle, loss_scaling=loss_scaling)

    bridge = ODBLightningBridge(
        module=module,
        handle=handle,
        sample_budget=sample_budget,
        loss_scaling=loss_scaling,
    )
    setattr(module, "odb_bridge", bridge)
    setattr(module, "odb_emitted_samples", 0)

    if wrap_training_step:
        if not hasattr(module, "training_step"):
            raise AttributeError("module has no training_step method to wrap")
        _wrap_training_step(module, bridge)

    return bridge


__all__ = [
    "ODBLightningBridge",
    "ODBLightningBridgeState",
    "ODBLightningCallback",
    "ODBLightningMixin",
    "configure_lightning_module",
]
