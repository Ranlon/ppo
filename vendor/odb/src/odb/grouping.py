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

"""ODB v5.1 grouping algorithm — local grouping + single all_gather + deterministic alignment."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import distributed as dist

from .constants import TOTAL_BATCH_SIZE_KEY, TOTAL_TOKENS_KEY
from .utils import compute_batch_size, get_input_length, get_n_patches

_GROUP_ORDER_FLIP_ALIASES = {
    "": "none",
    "0": "none",
    "false": "none",
    "none": "none",
    "a": "none",
    "rank_epoch_random": "rank_epoch_random",
    "epoch_random": "rank_epoch_random",
    "b": "rank_epoch_random",
    "rank_window_random": "rank_window_random",
    "window_random": "rank_window_random",
    "c": "rank_window_random",
    "rank_window_balanced": "rank_window_balanced",
    "window_balanced": "rank_window_balanced",
    "balanced": "rank_window_balanced",
    "d": "rank_window_balanced",
}


def normalize_group_order_flip(mode: str | None) -> str:
    """Normalize the rank-wise group-emission-order flip mode."""
    if mode is None:
        return "none"
    key = str(mode).strip().lower().replace("-", "_")
    try:
        return _GROUP_ORDER_FLIP_ALIASES[key]
    except KeyError as exc:
        valid = sorted({"none", "rank_epoch_random", "rank_window_random", "rank_window_balanced", "A", "B", "C", "D"})
        raise ValueError(f"group_order_flip must be one of {valid}, got {mode!r}.") from exc


def _stable_uint64(*parts: object) -> int:
    """Return a deterministic 64-bit hash independent of Python hash randomization."""
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), byteorder="big", signed=False)


def _rank_should_flip(mode: str | None, random_seed: int, grouping_round: int, rank: int, world_size: int) -> bool:
    """Whether a rank should emit its aligned groups in reverse order for this round."""
    mode = normalize_group_order_flip(mode)
    if mode == "none":
        return False
    if mode == "rank_epoch_random":
        return (_stable_uint64("rank_epoch_random", random_seed, rank) & 1) == 1
    if mode == "rank_window_random":
        return (_stable_uint64("rank_window_random", random_seed, grouping_round, rank) & 1) == 1

    # rank_window_balanced: approximately half the ranks flip each grouping
    # window, with a deterministic parity offset so the pattern can rotate.
    parity_offset = _stable_uint64("rank_window_balanced", random_seed, grouping_round, world_size) & 1
    return ((rank + parity_offset) & 1) == 1


def _rank_group_order_indices(
    n_groups: int,
    mode: str | None,
    random_seed: int,
    grouping_round: int,
    rank: int,
    world_size: int,
) -> list[int]:
    """Return post-alignment group indices for one rank."""
    order = list(range(n_groups))
    if n_groups <= 1:
        return order
    if _rank_should_flip(mode, random_seed, grouping_round, rank, world_size):
        order.reverse()
    return order


def _apply_group_order(values: list[Any], order: list[int]) -> list[Any]:
    """Reorder a per-rank group-sized list using precomputed indices."""
    if len(order) != len(values):
        raise ValueError(f"group-order length mismatch: order={len(order)} values={len(values)}")
    return [values[i] for i in order]


def grouping_data(
    data_buffer: list[list[dict]],
    max_input_length: int,
    ddp_group: Any,
    is_finished: bool = False,
    idx_budget: int = 0,
    join_mode: bool = False,
    loss_scaling: bool = False,
    loss_scaling_approx: bool = True,
    max_groups: int = 4096,
    max_patches: int = 0,
    group_order_flip: str | None = "none",
    random_seed: int = 1042,
    grouping_round: int = 0,
) -> tuple[list[list[dict]], list[dict], bool, bool]:
    """Group samples dynamically based on length with DDP synchronization.

    This is the core ODB v5.1 algorithm:

    1. Flatten and sort samples by length.
    2. Form local groups greedily (longest-first, split when batch full).
    3. In DDP mode: single ``all_gather`` to exchange group counts and sizes.
    4. Deterministic alignment: all ranks compute the same target count, then
       split or drop groups to match.  Overflow samples are returned for
       recycling.
    5. ``total_batch_size`` is set on **every** sample in each group.

    Args:
        data_buffer: Nested list of sample dicts from workers.
        max_input_length: Reference length for batch-size calculation.
        ddp_group: ``dist.ProcessGroup`` for DDP sync, or ``None``.
        is_finished: ``True`` when the underlying iterator is exhausted.
        idx_budget: Number of output slots available in the caller.
        join_mode: If ``True``, a finished rank does not force all ranks to
            stop immediately; only non-finished ranks participate in the
            output-budget predicate, and termination waits until every rank
            reports finished.
        loss_scaling: If ``True``, piggyback per-group token counts in the
            existing ``all_gather`` (zero extra communication) and inject
            ``odb_total_tokens`` into each sample for precise loss scaling.
        max_groups: Maximum number of groups per rank for the all_gather
            tensor.  Tied to ``buffer_size`` by the caller so that local
            groups never exceed the communication capacity.
        group_order_flip: Optional rank-wise post-alignment group-order flip
            mode. ``"none"`` preserves the existing order; ``"rank_epoch_random"``
            flips each rank once per iterator; ``"rank_window_random"`` samples a
            deterministic flip per grouping window and rank;
            ``"rank_window_balanced"`` flips roughly half the ranks per window.
        random_seed: Iterator seed used to make non-``none`` flip modes
            deterministic across ranks.
        grouping_round: Monotonic grouping-window counter from the collate loop.

    Returns:
        ``(grouped_data, overflow_samples, is_all_finished, skip_output)``

        - *grouped_data*: list of groups (each group is a list of sample dicts).
        - *overflow_samples*: samples that didn't fit and should be recycled.
        - *is_all_finished*: ``True`` when all ranks have finished.
        - *skip_output*: ``True`` when the caller should skip sending output
          (e.g. waiting for other ranks to catch up).
    """

    # ---- helpers ----
    def _group_by_length(sorted_data: list[dict]) -> list[list[dict]]:
        """Greedily form groups from longest to shortest."""
        groups: list[list[dict]] = []
        batch_size = 0
        batch: list[dict] = []
        batch_patches = 0

        for sample in reversed(sorted_data):
            length = get_input_length(sample)
            n_p = get_n_patches(sample)

            if max_patches > 0 and batch and (batch_patches + n_p) > max_patches:
                groups.insert(0, batch)
                batch = []
                batch_patches = 0

            batch.insert(0, sample)
            batch_patches += n_p
            if len(batch) >= batch_size:
                batch_size = compute_batch_size(length, max_input_length, n_p, max_patches)
                if batch:
                    groups.insert(0, batch)
                batch = []
                batch_patches = 0

        if batch:
            groups.insert(0, batch)

        return groups

    def _deterministic_align_sizes(
        group_sizes: list[int], target: int
    ) -> tuple[list[int], list[int]]:
        """Align the number of groups to *target* deterministically.

        If fewer groups than target: split the largest group.
        If more groups than target: keep the largest, overflow the rest.
        """
        if len(group_sizes) == 0:
            return [], []

        if len(group_sizes) == target:
            return list(group_sizes), []

        if len(group_sizes) < target:
            aligned = list(group_sizes)
            i = len(aligned) - 1
            while len(aligned) < target:
                if i < 0:
                    break
                if aligned[i] < 2:
                    i -= 1
                else:
                    aligned[i] -= 1
                    aligned.insert(i + 1, 1)
            return aligned, []

        # More groups than target — keep the largest, overflow the rest
        indexed = sorted(range(len(group_sizes)), key=lambda j: group_sizes[j], reverse=True)
        keep_set = set(indexed[:target])
        aligned = [group_sizes[j] for j in range(len(group_sizes)) if j in keep_set]
        overflow = [group_sizes[j] for j in range(len(group_sizes)) if j not in keep_set]
        return aligned, overflow

    # ---- flatten & sort ----
    data_buffer_flat = sum(data_buffer, [])
    sorted_data = [d for d in data_buffer_flat if get_input_length(d) > 0]
    sorted_data = sorted(sorted_data, key=get_input_length)

    # ---- local grouping ----
    local_groups = _group_by_length(sorted_data)

    # ---- clamp to max_groups (overflow fallback) ----
    # If group count exceeds max_groups, excess groups' samples are recycled
    # to the next iteration.  With max_groups=4096 this is extremely unlikely,
    # but prevents the all_gather tensor from being under-filled while
    # local_n_groups advertises a larger count (which causes alignment bugs).
    clamped_overflow: list[dict] = []
    if len(local_groups) > max_groups:
        import logging

        logging.getLogger(__name__).warning(
            "ODB: local group count (%d) exceeds max_groups (%d), "
            "overflowing %d groups to next iteration",
            len(local_groups),
            max_groups,
            len(local_groups) - max_groups,
        )
        for g in local_groups[max_groups:]:
            clamped_overflow.extend(g)
        local_groups = local_groups[:max_groups]

    local_n_groups = -1 if is_finished else len(local_groups)

    # ---- DDP synchronization ----
    if ddp_group is not None:
        world_size = dist.get_world_size(ddp_group)
        my_rank = dist.get_rank(ddp_group)

        # Layout depends on loss_scaling:
        #   off: [idx_budget, n_groups, size_0..size_{MAX-1}]
        #   on:  [idx_budget, n_groups, size_0..size_{MAX-1}, total_tokens_0..total_tokens_{MAX-1}]
        SLOTS = 2 + max_groups * (2 if loss_scaling else 1)

        local_info = torch.zeros(SLOTS, dtype=torch.long, device="cpu")
        local_info[0] = idx_budget
        local_info[1] = local_n_groups
        token_offset = 2 + max_groups
        for i, group in enumerate(local_groups[:max_groups]):
            local_info[2 + i] = len(group)
            if loss_scaling:
                local_info[token_offset + i] = sum(get_input_length(s) for s in group)

        all_info = torch.zeros(world_size * SLOTS, dtype=torch.long, device="cpu")
        dist.all_gather_into_tensor(all_info, local_info, group=ddp_group)

        all_info = all_info.view(world_size, SLOTS)
        all_idx_budgets = all_info[:, 0]
        all_n_groups = all_info[:, 1]
        all_group_sizes = all_info[:, 2:2 + max_groups]
        all_group_tokens = all_info[:, token_offset:token_offset + max_groups] if loss_scaling else None

        # Determine finished / skip state.
        #
        # Non-join keeps the historical shortest-rank closure: once any rank
        # advertises finished, every rank exits the ODB epoch.  Join mode is
        # stricter: exhausted ranks stay in this Gloo protocol until all ranks
        # are finished, but the budget predicate ignores those finished ranks
        # because their trainer iterator has already stopped yielding batches.
        if join_mode:
            is_all_finished = bool((all_n_groups == -1).all().item())
            active_or_empty_mask = all_n_groups >= 0
            if active_or_empty_mask.any():
                skip_output = bool(all_idx_budgets[active_or_empty_mask].min().item() == 0)
            else:
                skip_output = False
        else:
            any_finished = bool((all_n_groups == -1).any().item())
            if any_finished:
                is_all_finished = True
                skip_output = False
            else:
                is_all_finished = False
                skip_output = bool(all_idx_budgets.min().item() == 0)

        if skip_output:
            return [], list(sorted_data), is_all_finished, True

        active_mask = all_n_groups > 0
        if not active_mask.any():
            return [], [], is_all_finished, False

        active_groups = all_n_groups[active_mask]
        active_budgets = all_idx_budgets[active_mask]
        positive_budgets = active_budgets[active_budgets > 0]
        active_n_samples = all_group_sizes[active_mask].sum(dim=1)
        positive_n_samples = active_n_samples[active_n_samples > 0]

        target = int(active_groups.max().item())
        if len(positive_budgets) > 0:
            target = min(target, int(positive_budgets.min().item()))
        if len(positive_n_samples) > 0:
            target = min(target, int(positive_n_samples.min().item()))
        target = max(target, 1)

        # Deterministic alignment: all ranks simulate all ranks
        all_aligned_sizes: list[list[int]] = []
        for r in range(world_size):
            n_g = int(all_n_groups[r].item())
            if n_g <= 0:
                all_aligned_sizes.append([])
                continue
            gs = all_group_sizes[r, :n_g].tolist()
            aligned_s, _ = _deterministic_align_sizes(gs, target)
            all_aligned_sizes.append(aligned_s)

        # Apply alignment to local groups
        my_n_g = int(all_n_groups[my_rank].item())
        if my_n_g <= 0:
            return [], list(sorted_data), is_all_finished, False

        if len(local_groups) < target:
            while len(local_groups) < target:
                split_done = False
                for i in range(len(local_groups) - 1, -1, -1):
                    if len(local_groups[i]) >= 2:
                        split_sample = local_groups[i].pop(-1)
                        local_groups.insert(i + 1, [split_sample])
                        split_done = True
                        break
                if not split_done:
                    break
            grouped_data = local_groups[:target]
            overflow_samples: list[dict] = []
            for g in local_groups[target:]:
                overflow_samples.extend(g)
        elif len(local_groups) > target:
            indexed = sorted(range(len(local_groups)), key=lambda j: len(local_groups[j]), reverse=True)
            keep_set = set(indexed[:target])
            grouped_data = [local_groups[j] for j in range(len(local_groups)) if j in keep_set]
            overflow_samples = []
            for j in range(len(local_groups)):
                if j not in keep_set:
                    overflow_samples.extend(local_groups[j])
        else:
            grouped_data = local_groups
            overflow_samples = []

        # Optional rank-wise post-alignment group-order flip.  This must happen
        # before total_batch_size / token-loss metadata injection so that every
        # rank computes metadata for the same cross-rank emission position.
        rank_orders = [
            _rank_group_order_indices(
                len(all_aligned_sizes[r]),
                group_order_flip,
                random_seed,
                grouping_round,
                r,
                world_size,
            )
            for r in range(world_size)
        ]
        # Keep the pre-flip aligned sizes for token-approximation.  Approx mode
        # estimates tokens in original alignment positions, then reorders the
        # token vector to match post-flip emission positions.
        all_aligned_sizes_for_tokens = [list(sizes) for sizes in all_aligned_sizes]
        all_aligned_sizes = [
            _apply_group_order(all_aligned_sizes[r], rank_orders[r])
            for r in range(world_size)
        ]
        grouped_data = _apply_group_order(grouped_data, rank_orders[my_rank])

        # ---- Loss scaling: compute total_tokens per group across all ranks ----
        if loss_scaling and all_group_tokens is not None:
            # Check if ANY rank had adjustment (deterministic — all ranks see same all_n_groups & target)
            any_rank_adjusted = any(
                int(all_n_groups[r].item()) != target
                for r in range(world_size)
                if int(all_n_groups[r].item()) > 0
            )

            if not any_rank_adjusted:
                # No adjustment on any rank: first all_gather data is exact.
                # Clone because non-``none`` group-order modes may reorder the
                # per-rank prefix to match post-flip emission positions.
                all_total_tokens = all_group_tokens.clone()  # shape: [world_size, max_groups]
            elif loss_scaling_approx:
                # Approximate mode: use avg_tlen from first all_gather to estimate
                all_total_tokens = torch.zeros(world_size, max_groups, dtype=torch.long)
                for r in range(world_size):
                    n_g = int(all_n_groups[r].item())
                    if n_g <= 0 or len(all_aligned_sizes_for_tokens[r]) == 0:
                        continue
                    for i in range(min(target, len(all_aligned_sizes_for_tokens[r]))):
                        adjusted_size = all_aligned_sizes_for_tokens[r][i]
                        orig_idx = min(i, n_g - 1)
                        orig_size = int(all_group_sizes[r][orig_idx].item())
                        orig_tokens = int(all_group_tokens[r][orig_idx].item())
                        if orig_size > 0:
                            avg_tlen = orig_tokens / orig_size
                            all_total_tokens[r][i] = int(adjusted_size * avg_tlen)
                        else:
                            all_total_tokens[r][i] = 0
            else:
                # Exact mode: second all_gather with post-alignment token counts
                local_tokens_info = torch.zeros(max_groups, dtype=torch.long, device="cpu")
                for i, group in enumerate(grouped_data[:max_groups]):
                    local_tokens_info[i] = sum(get_input_length(s) for s in group)

                all_tokens_info = torch.zeros(world_size * max_groups, dtype=torch.long, device="cpu")
                dist.all_gather_into_tensor(all_tokens_info, local_tokens_info, group=ddp_group)
                all_total_tokens = all_tokens_info.view(world_size, max_groups)

            if normalize_group_order_flip(group_order_flip) != "none" and (
                not any_rank_adjusted or loss_scaling_approx
            ):
                reordered_tokens = all_total_tokens.clone()
                for r, order in enumerate(rank_orders):
                    if order:
                        index = torch.tensor(order, dtype=torch.long, device=all_total_tokens.device)
                        reordered_tokens[r, : len(order)] = all_total_tokens[r, index]
                all_total_tokens = reordered_tokens

        # Set total_batch_size and odb_total_tokens on ALL samples in each group
        for i, group in enumerate(grouped_data):
            total_bs = sum(
                all_aligned_sizes[r][i]
                for r in range(world_size)
                if i < len(all_aligned_sizes[r])
            )
            for sample in group:
                sample[TOTAL_BATCH_SIZE_KEY] = total_bs

            if loss_scaling and all_group_tokens is not None:
                total_tokens = sum(
                    int(all_total_tokens[r][i].item())
                    for r in range(world_size)
                    if i < len(all_aligned_sizes[r])
                )
                for sample in group:
                    sample[TOTAL_TOKENS_KEY] = total_tokens

        overflow_samples.extend(clamped_overflow)
        return grouped_data, overflow_samples, is_all_finished, False
    else:
        # ---- Non-DDP mode ----
        is_all_finished = is_finished

        if len(local_groups) == 0:
            return [], [], is_all_finished, False

        target = len(local_groups)
        if idx_budget > 0:
            target = min(target, idx_budget)
        target = max(target, 1)

        if len(local_groups) > target:
            indexed = sorted(range(len(local_groups)), key=lambda j: len(local_groups[j]), reverse=True)
            keep_set = set(indexed[:target])
            grouped_data = [local_groups[j] for j in range(len(local_groups)) if j in keep_set]
            overflow_samples = []
            for j in range(len(local_groups)):
                if j not in keep_set:
                    overflow_samples.extend(local_groups[j])
        else:
            grouped_data = local_groups[:target]
            overflow_samples = []
            for g in local_groups[target:]:
                overflow_samples.extend(g)

        order = _rank_group_order_indices(
            len(grouped_data),
            group_order_flip,
            random_seed,
            grouping_round,
            rank=0,
            world_size=1,
        )
        grouped_data = _apply_group_order(grouped_data, order)

        # Set total_batch_size on ALL samples in each group
        for group in grouped_data:
            total_tokens = sum(get_input_length(s) for s in group) if loss_scaling else None
            for sample in group:
                sample[TOTAL_BATCH_SIZE_KEY] = len(group)
                if total_tokens is not None:
                    sample[TOTAL_TOKENS_KEY] = total_tokens

        overflow_samples.extend(clamped_overflow)
        return grouped_data, overflow_samples, is_all_finished, False
