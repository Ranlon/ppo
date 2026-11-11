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

"""Upward runtime information from ODB batches to trainer integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import normalize_loss_scaling
from .constants import (
    LOCAL_BATCH_SIZE_KEY,
    LOCAL_TOKENS_KEY,
    ODB_STEP_INFO_KEY,
    TOTAL_BATCH_SIZE_KEY,
    TOTAL_TOKENS_KEY,
)


@dataclass(frozen=True)
class ODBStepInfo:
    """Minimal trainer-facing runtime info for one yielded ODB batch."""

    all_samples_this_step: int
    loss_scale: float | torch.Tensor = 1.0


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _to_int(value[0], default=default)
    return int(value)


def _infer_local_samples(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 0


def _world_size() -> int:
    try:
        from torch import distributed as dist

        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    except Exception:
        return 1


def pop_step_info(
    batch: dict[str, Any],
    *,
    loss_scaling: bool | str | None = "auto",
    world_size: int | None = None,
) -> ODBStepInfo:
    """Remove ODB transport metadata and return clean trainer-facing step info.

    ``loss_scaling="auto"`` preserves legacy behavior for direct calls: token
    metadata is preferred when present, otherwise sample metadata is used.
    Trainer adapters should pass the resolved config mode explicitly so
    ``loss_scaling="none"`` returns ``loss_scale=1.0``.
    """
    existing = batch.pop(ODB_STEP_INFO_KEY, None)
    if isinstance(existing, ODBStepInfo):
        _pop_legacy_transport(batch)
        return existing

    total_samples_raw = batch.pop(TOTAL_BATCH_SIZE_KEY, None)
    local_samples_raw = batch.pop(LOCAL_BATCH_SIZE_KEY, None)
    local_tokens_raw = batch.pop(LOCAL_TOKENS_KEY, None)
    total_tokens_raw = batch.pop(TOTAL_TOKENS_KEY, None)

    local_samples = _to_int(local_samples_raw, default=_infer_local_samples(batch))
    all_samples = _to_int(total_samples_raw, default=local_samples)
    local_tokens = _to_int(local_tokens_raw, default=0)
    total_tokens = _to_int(total_tokens_raw, default=0)

    mode = "auto" if loss_scaling is None else str(loss_scaling).lower().replace("-", "_")
    if mode != "auto":
        mode = normalize_loss_scaling(loss_scaling)

    scale_basis_local = 0
    scale_basis_total = 0
    if mode in {"approx", "exact"}:
        scale_basis_local = local_tokens
        scale_basis_total = total_tokens
    elif mode == "none":
        scale_basis_local = 0
        scale_basis_total = 0
    else:
        if local_tokens and total_tokens:
            scale_basis_local = local_tokens
            scale_basis_total = total_tokens
        elif local_samples and all_samples:
            scale_basis_local = local_samples
            scale_basis_total = all_samples

    if scale_basis_local and scale_basis_total:
        ws = _world_size() if world_size is None else int(world_size)
        loss_scale: float | torch.Tensor = (scale_basis_local / scale_basis_total) * ws
    else:
        loss_scale = 1.0

    return ODBStepInfo(all_samples_this_step=all_samples, loss_scale=loss_scale)


def _pop_legacy_transport(batch: dict[str, Any]) -> None:
    batch.pop(TOTAL_BATCH_SIZE_KEY, None)
    batch.pop(LOCAL_BATCH_SIZE_KEY, None)
    batch.pop(LOCAL_TOKENS_KEY, None)
    batch.pop(TOTAL_TOKENS_KEY, None)
