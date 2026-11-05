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

"""Utility functions for ODB."""

from __future__ import annotations

from typing import Any

import torch


def null_collate_fn(batch: list[Any]) -> list[Any]:
    """Identity collate function — passes samples through without modification.

    Used in ODB worker processes so that raw samples reach the collate process
    without being batched or transformed by the default collator.
    """
    return batch


class _IDInjectingDataset:
    """Wraps a dataset to inject ``_odb_sample_idx`` into each item dict.

    Used only when the env var ``ODB_LOG_EMITTED_IDS`` is set, to enable
    per-sample identity-coverage auditing in :mod:`odb.collate`.  Items
    that are not ``dict``-like are passed through unchanged.
    """

    def __init__(self, dataset: Any) -> None:
        self._odb_dataset = dataset

    def __len__(self) -> int:
        return len(self._odb_dataset)

    def __getitem__(self, idx: int) -> Any:
        item = self._odb_dataset[idx]
        if isinstance(item, dict):
            item["_odb_sample_idx"] = int(idx)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._odb_dataset, name)


def get_input_length(data: dict) -> int:
    """Get the sequence length from a sample dict.

    Handles both tensor formats (``[1, seq_len]`` or ``[seq_len]``) and plain
    Python lists.

    Args:
        data: A sample dict containing an ``"input_ids"`` key.

    Returns:
        The sequence length, or 0 if input_ids is missing/empty.
    """
    input_ids = data.get("input_ids")
    if input_ids is None:
        return 0
    if isinstance(input_ids, torch.Tensor):
        if input_ids.numel() == 0:
            return 0
        if input_ids.dim() == 2:
            return input_ids.size(1)
        elif input_ids.dim() == 1:
            return input_ids.size(0)
        else:
            return 0
    elif isinstance(input_ids, (list, tuple)):
        return len(input_ids)
    else:
        return 0


def get_n_patches(data: dict) -> int:
    """Get the number of ViT patches from a sample dict.

    Returns 0 if the field is absent (text-only sample).
    """
    val = data.get("odb_n_patches", 0)
    if isinstance(val, torch.Tensor):
        return int(val.item())
    return int(val) if val else 0


def is_valid_batch(batch: list[dict]) -> bool:
    """Check whether a batch contains valid (non-empty) data.

    Returns ``False`` if *any* sample has missing or zero-length input_ids.
    """
    if not batch:
        return False
    for d in batch:
        input_ids = d.get("input_ids")
        if input_ids is None:
            return False
        if isinstance(input_ids, torch.Tensor):
            if input_ids.numel() == 0:
                return False
            if input_ids.dim() == 2:
                if input_ids.size(1) == 0:
                    return False
            elif input_ids.dim() == 1:
                if input_ids.size(0) == 0:
                    return False
            else:
                return False
        elif isinstance(input_ids, (list, tuple)):
            if len(input_ids) == 0:
                return False
        else:
            return False
    return True


def compute_batch_size(
    input_length: int,
    max_input_length: int,
    n_patches: int = 0,
    max_patches: int = 0,
) -> int:
    """Compute the target batch size for a given input length.

    Longer sequences get smaller batches so that the total token count per
    batch stays roughly constant::

        batch_size = max(max_input_length // input_length, 1)

    When *n_patches* and *max_patches* are both positive, the batch size is
    further clamped so that the total vision patches per batch does not exceed
    *max_patches*.  This prevents ViT OOM on image-heavy batches without
    penalising LLM throughput on text-only data.

    Args:
        input_length: The sequence length of the current sample.
        max_input_length: The reference maximum length (e.g. 16384).
        n_patches: Number of ViT patches in this sample (0 for text-only).
        max_patches: Maximum total patches per batch (0 to disable).

    Returns:
        Target batch size (>= 1).
    """
    if input_length == 0:
        return 1
    bs = int(max(max_input_length / input_length, 1))
    if n_patches > 0 and max_patches > 0:
        bs = min(bs, max(max_patches // n_patches, 1))
    return bs


def scale_loss(
    loss: torch.Tensor,
    batch_or_local_bs: int | dict = 0,
    total_batch_size: int = 0,
    world_size: int = 0,
) -> torch.Tensor:
    """Scale loss for correct gradient averaging with variable batch sizes.

    In ODB, different ranks may have different local batch sizes for the same
    group.  Standard DDP all-reduce computes ``mean(grad_r)`` across ranks,
    but ``grad_r = sum(per_sample_grads) / local_bs_r``, which gives unequal
    weight to ranks with smaller batches.

    This function scales the loss so that after ``all_reduce(mean)``, the
    result equals the true per-sample mean over all ranks::

        scaled_loss = loss * (local_bs / total_bs) * world_size

    Can be called in two ways:

    1. **With a batch dict** (recommended)::

           scaled_loss = odb.scale_loss(loss, batch)

       Automatically extracts ``local_batch_size``, ``total_batch_size``
       from the batch dict, and infers ``world_size`` from
       ``torch.distributed``.

    2. **With explicit values**::

           scaled_loss = odb.scale_loss(loss, local_bs, total_bs, world_size)

    Args:
        loss: The unscaled per-sample mean loss on this rank.
        batch_or_local_bs: Either a batch dict (containing ODB metadata keys)
            or an integer ``local_batch_size``.
        total_batch_size: Sum of local batch sizes across all ranks.
            Ignored when *batch_or_local_bs* is a dict.
        world_size: Number of DDP / FSDP ranks.
            Ignored when *batch_or_local_bs* is a dict.

    Returns:
        Scaled loss tensor (same device, differentiable).
    """
    if isinstance(batch_or_local_bs, dict):
        batch = batch_or_local_bs
        from .constants import LOCAL_BATCH_SIZE_KEY, LOCAL_TOKENS_KEY, TOTAL_BATCH_SIZE_KEY, TOTAL_TOKENS_KEY

        # Prefer token-level scaling (more precise) if available
        local_tokens = batch.get(LOCAL_TOKENS_KEY)
        total_tokens = batch.get(TOTAL_TOKENS_KEY)
        if local_tokens is not None and total_tokens is not None:
            if isinstance(local_tokens, torch.Tensor):
                local_tokens = int(local_tokens.item())
            if isinstance(total_tokens, torch.Tensor):
                total_tokens = int(total_tokens.item())
            local_batch_size = int(local_tokens)
            total_batch_size = int(total_tokens)
        else:
            # Fall back to sample-level scaling
            local_batch_size = batch.get(LOCAL_BATCH_SIZE_KEY, 0)
            total_batch_size = batch.get(TOTAL_BATCH_SIZE_KEY, 0)
            if isinstance(local_batch_size, torch.Tensor):
                local_batch_size = int(local_batch_size.item())
            if isinstance(total_batch_size, torch.Tensor):
                total_batch_size = int(total_batch_size.item())
        try:
            from torch import distributed as dist

            world_size = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            world_size = 1
    else:
        local_batch_size = int(batch_or_local_bs)

    if total_batch_size == 0 or world_size == 0:
        return loss
    return loss * (local_batch_size / total_batch_size) * world_size
