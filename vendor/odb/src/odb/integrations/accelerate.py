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

"""HuggingFace Accelerate adapter for ODB custom training loops."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from odb.core import apply
from odb.handle import ODBHandle
from odb.step_info import ODBStepInfo, pop_step_info

if TYPE_CHECKING:
    from collections.abc import ContextManager, Iterable

    from torch.utils.data import DataLoader


@dataclass
class ODBAccelerateBridgeState:
    emitted_samples: int = 0
    optimizer_steps: int = 0
    last_step_info: ODBStepInfo | None = None


@dataclass
class ODBAccelerateBridge:
    """Small runtime bridge for Accelerate-based custom loops.

    Accelerate intentionally keeps the training loop in user code, so ODB does
    the same: call :meth:`consume_batch` before ``model(**batch)`` and either
    pass the returned info to :meth:`scale_loss` or use :meth:`backward`.
    """

    accelerator: Any
    dataloader: "DataLoader"
    handle: ODBHandle
    sample_budget: int | None = None
    loss_scaling: bool | str | None = None
    state: ODBAccelerateBridgeState = field(default_factory=ODBAccelerateBridgeState)

    def __post_init__(self) -> None:
        if self.sample_budget is not None and self.sample_budget <= 0:
            raise ValueError(f"sample_budget must be > 0, got {self.sample_budget}")
        if self.loss_scaling is None:
            self.loss_scaling = self.handle.config.loss_scaling

    def consume_batch(self, batch: dict[str, Any]) -> ODBStepInfo:
        """Pop ODB metadata from ``batch`` and update emitted-sample progress."""
        info = pop_step_info(batch, loss_scaling=self.loss_scaling)
        self.state.last_step_info = info
        self.state.emitted_samples += int(info.all_samples_this_step)
        return info

    def scale_loss(self, loss, info: ODBStepInfo | None = None):
        """Apply the ODB loss multiplier for the current micro-step."""
        step_info = info if info is not None else self.state.last_step_info
        if step_info is None:
            raise RuntimeError("No ODB step info is available. Call consume_batch(batch) before scaling loss.")
        return loss * step_info.loss_scale

    def backward(self, loss, info: ODBStepInfo | None = None, **kwargs):
        """Scale ``loss`` and delegate to ``accelerator.backward``."""
        scaled_loss = self.scale_loss(loss, info=info)
        self.accelerator.backward(scaled_loss, **kwargs)
        return scaled_loss

    def mark_optimizer_step(self) -> None:
        """Record one optimizer update in custom loops that want this counter."""
        self.state.optimizer_steps += 1

    @property
    def should_stop(self) -> bool:
        return self.sample_budget is not None and self.state.emitted_samples >= self.sample_budget

    def join_uneven_inputs(self, joinables: "Iterable[Any]", **kwargs) -> "ContextManager[Any]":
        """Return Accelerate's uneven-input context when available.

        ODB's join mode drains the DataLoader/collate side. DDP model
        collectives still need the framework's uneven-input protection.
        """
        join_fn = getattr(self.accelerator, "join_uneven_inputs", None)
        if join_fn is None:
            return nullcontext()
        return join_fn(joinables, **kwargs)


def configure_accelerator(
    accelerator: Any,
    dataloader: "DataLoader",
    *,
    handle: ODBHandle | None = None,
    sample_budget: int | None = None,
    prepare_dataloader: bool = False,
    loss_scaling: bool | str | None = None,
    **apply_kwargs,
) -> ODBAccelerateBridge:
    """Configure an Accelerate loop to consume an ODB dataloader.

    ``dataloader`` should normally be ODB-enabled before
    ``accelerator.prepare(...)``.  Set ``prepare_dataloader=True`` when you want
    this helper to call ``accelerator.prepare`` for the dataloader and store the
    prepared loader in the returned bridge.
    """
    if handle is None:
        handle = apply(dataloader, **apply_kwargs)
    elif apply_kwargs:
        raise ValueError("Do not pass ODB apply arguments when handle is already provided.")

    prepared_dataloader = accelerator.prepare(dataloader) if prepare_dataloader else dataloader
    return ODBAccelerateBridge(
        accelerator=accelerator,
        dataloader=prepared_dataloader,
        handle=handle,
        sample_budget=sample_budget,
        loss_scaling=loss_scaling,
    )


__all__ = [
    "ODBAccelerateBridge",
    "ODBAccelerateBridgeState",
    "configure_accelerator",
]
