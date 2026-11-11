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

"""ODB collate loop — runs in a dedicated process, handles overflow recycling."""

from __future__ import annotations

from collections.abc import MutableMapping
import datetime
import json
import logging
import multiprocessing
import os
import queue
import random
import time
from typing import Any, Callable

import torch
from torch import distributed as dist
from torch.utils.data.dataloader import _utils

from .constants import (
    IDLE_DATA,
    LOCAL_BATCH_SIZE_KEY,
    LOCAL_TOKENS_KEY,
    TOTAL_BATCH_SIZE_KEY,
    TOTAL_TOKENS_KEY,
)
from .grouping import grouping_data
from .utils import get_input_length, is_valid_batch

logger = logging.getLogger(__name__)


def _collate_group(batch: list[dict], collate_fn: Callable, loss_scaling: bool) -> Any:
    """Keep ODB transport fields outside framework collators.

    Hugging Face collators may discard unknown sample keys. Read the all-rank
    metadata before calling them, then attach it to their returned batch.
    """
    total_batch_size = batch[0].pop(TOTAL_BATCH_SIZE_KEY, None)
    total_tokens = batch[0].pop(TOTAL_TOKENS_KEY, None)
    if total_batch_size is None or (loss_scaling and total_tokens is None):
        raise ValueError("ODB grouped sample is missing global batch metadata")
    for sample in batch[1:]:
        if sample.pop(TOTAL_BATCH_SIZE_KEY, None) != total_batch_size:
            raise ValueError("ODB group has inconsistent global batch size")
        if sample.pop(TOTAL_TOKENS_KEY, None) != total_tokens:
            raise ValueError("ODB group has inconsistent global token count")

    local_tokens = sum(get_input_length(sample) for sample in batch) if loss_scaling else None
    data = collate_fn(batch)
    if isinstance(data, MutableMapping) and data:
        data[TOTAL_BATCH_SIZE_KEY] = total_batch_size
        data[LOCAL_BATCH_SIZE_KEY] = len(batch)
        if loss_scaling:
            data[TOTAL_TOKENS_KEY] = total_tokens
            data[LOCAL_TOKENS_KEY] = local_tokens
    return data


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except ValueError:
        return default


def _configure_gloo_socket_ifname() -> str:
    gloo_ifname = os.environ.get("GLOO_SOCKET_IFNAME", "").strip()
    if gloo_ifname:
        return gloo_ifname

    odb_ifname = os.environ.get("ODB_GLOO_SOCKET_IFNAME", "").strip()
    if odb_ifname:
        os.environ["GLOO_SOCKET_IFNAME"] = odb_ifname
        return odb_ifname

    nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "").strip()
    if nccl_ifname and "^" not in nccl_ifname:
        os.environ["GLOO_SOCKET_IFNAME"] = nccl_ifname
        return nccl_ifname
    return ""


def _create_odb_gloo_group(random_seed: int):
    iface = _configure_gloo_socket_ifname()
    rank = dist.get_rank()
    retries = _env_int("ODB_GROUP_INIT_RETRIES", 4)
    timeout_s = _env_int("ODB_GROUP_INIT_TIMEOUT_SEC", 60)
    base_sleep = _env_float("ODB_GROUP_INIT_RETRY_BASE_SEC", 5.0)
    max_sleep = _env_float("ODB_GROUP_INIT_RETRY_MAX_SEC", 20.0)
    log_init = os.environ.get("ODB_LOG_GROUP_INIT", "0") == "1"

    world = getattr(dist.distributed_c10d, "_world", None)
    original_group_count = getattr(world, "group_count", None)

    for attempt in range(retries):
        try:
            if original_group_count is not None:
                world.group_count = original_group_count + random_seed

            group = dist.new_group(
                backend="gloo",
                timeout=datetime.timedelta(seconds=timeout_s),
            )

            if original_group_count is not None:
                world.group_count = original_group_count + 1

            if log_init:
                suffix = f", GLOO_SOCKET_IFNAME={iface}" if iface else ""
                logger.info(
                    f"[ODB][rank {rank}] created Gloo metadata group "
                    f"(attempt {attempt + 1}/{retries}, timeout={timeout_s}s{suffix})"
                )
            return group
        except Exception as exc:
            if original_group_count is not None:
                world.group_count = original_group_count
            if attempt >= retries - 1:
                logger.exception(
                    f"[ODB][rank {rank}] dist.new_group(backend='gloo') failed "
                    f"after {retries} attempts "
                    f"(timeout={timeout_s}s, GLOO_SOCKET_IFNAME={iface or '<auto>'})"
                )
                raise

            wait = min(max_sleep, base_sleep * (2 ** attempt)) + random.random()
            logger.warning(
                f"[ODB][rank {rank}] dist.new_group(backend='gloo') failed "
                f"(attempt {attempt + 1}/{retries}, timeout={timeout_s}s, "
                f"GLOO_SOCKET_IFNAME={iface or '<auto>'}): {exc}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    raise RuntimeError("unreachable ODB Gloo group initialization state")


def collate_loop_odb(
    in_queue: multiprocessing.Queue,
    out_queue: multiprocessing.Queue,
    collate_fn: Callable,
    buffer_size: int,
    max_input_length: int,
    done_event: multiprocessing.Event,
    random_seed: int = 1042,
    loss_scaling: bool = False,
    loss_scaling_approx: bool = True,
    join_mode: bool = False,
    no_warmup: bool = False,
    no_sync: bool = False,
    max_patches: int = 0,
    group_order_flip: str | None = "none",
) -> None:
    """Collate loop for ODB DataLoader with DDP synchronization.

    Runs in a separate process and:

    1. Collects samples from workers into a buffer.
    2. Creates a new DDP process group for synchronization (if DDP is active).
    3. Calls :func:`grouping_data` which uses ``all_gather`` to sync lengths.
    4. Recycles overflow samples into the next round.
    5. Sends collated batches to the output queue.

    Args:
        in_queue: Queue receiving ``(idx, data)`` tuples from workers.
        out_queue: Queue for sending collated batches to the main process.
        collate_fn: The user's collate function.
        buffer_size: Number of samples to collect before triggering grouping.
        max_input_length: Reference length for batch-size calculation.
        done_event: Event signalling that the main process wants to shut down.
        random_seed: Seed for deterministic shuffling of groups.
        join_mode: If ``True``, exhausted ranks keep the collate process alive
            for ODB's cross-rank grouping protocol while the main iterator stops
            yielding batches on that rank.
        group_order_flip: Optional post-alignment rank-wise group-order flip
            mode forwarded to :func:`grouping_data`.
    """
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(1)

    # Create a new process group for synchronization (with retry for timing)
    if no_sync:
        ddp_group = None
    elif dist.is_initialized():
        ddp_group = _create_odb_gloo_group(random_seed)
    else:
        ddp_group = None

    rg = random.Random(random_seed)

    # ------------------------------------------------------------------
    # Optional: per-emit sample-ID logging for identity-coverage audit.
    # Enabled by env var ODB_LOG_EMITTED_IDS=<path-template> where the
    # template may contain "{RANK}" which is substituted with the current
    # rank. One JSONL line per emitted batch:
    #   {"step": <int>, "ids": [<int>, ...]}
    # The dataset must inject "_odb_sample_idx" into each sample dict
    # (see _IDInjectingDataset in utils.py); samples without the key are
    # skipped silently.
    # ------------------------------------------------------------------
    _id_log_path: str | None = None
    _id_log_step = [0]
    _id_log_template = os.environ.get("ODB_LOG_EMITTED_IDS")
    if _id_log_template:
        _rank = dist.get_rank() if dist.is_initialized() else 0
        _id_log_path = _id_log_template.replace("{RANK}", str(_rank))
        _id_log_dir = os.path.dirname(_id_log_path)
        if _id_log_dir:
            os.makedirs(_id_log_dir, exist_ok=True)
        # Truncate any previous content so the file represents this run only.
        with open(_id_log_path, "w") as _f:
            _f.write("")
        logger.info("[ODB] emitted-IDs log -> %s", _id_log_path)

    def _log_emitted_ids(batch: list) -> None:
        if _id_log_path is None or not batch:
            return
        ids: list[int] = []
        for s in batch:
            if isinstance(s, dict) and "_odb_sample_idx" in s:
                ids.append(int(s.pop("_odb_sample_idx")))
        if not ids:
            return
        with open(_id_log_path, "a") as f:
            f.write(json.dumps({"step": _id_log_step[0], "ids": ids}) + "\n")
        _id_log_step[0] += 1

    # ------------------------------------------------------------------
    # collate_once — process one buffer of samples with overflow recycling
    # ------------------------------------------------------------------
    grouping_round = 0

    def collate_once(
        idx_buffer: list[int],
        data_buffer: list[list],
        is_finished: bool = False,
    ) -> bool:
        nonlocal grouping_round
        idx_budget = len(idx_buffer)
        current_grouping_round = grouping_round
        grouping_round += 1

        grouped_data, overflow, is_all_finished, skip_output = grouping_data(
            data_buffer,
            max_input_length,
            ddp_group,
            is_finished=is_finished,
            idx_budget=idx_budget,
            join_mode=join_mode,
            loss_scaling=loss_scaling,
            loss_scaling_approx=loss_scaling_approx,
            max_groups=buffer_size,
            max_patches=max_patches,
            group_order_flip=group_order_flip,
            random_seed=random_seed,
            grouping_round=current_grouping_round,
        )

        data_buffer.clear()
        if overflow:
            data_buffer.extend([[s] for s in overflow])

        if skip_output:
            return is_all_finished

        if join_mode and is_finished and idx_budget == 0:
            return is_all_finished

        rg.shuffle(grouped_data)
        sorted_idx = sorted(idx_buffer)

        batch_idx = 0
        while len(sorted_idx) > 0:
            idx = sorted_idx.pop(0)

            if batch_idx < len(grouped_data):
                batch = grouped_data[batch_idx]
                batch_idx += 1
                if is_valid_batch(batch):
                    _log_emitted_ids(batch)
                    data = _collate_group(batch, collate_fn, loss_scaling)
                    if isinstance(data, dict) and not data:
                        data = IDLE_DATA
                else:
                    data = IDLE_DATA
            else:
                data = IDLE_DATA

            while not done_event.is_set():
                try:
                    out_queue.put((idx, data), timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    break
                except queue.Full:
                    time.sleep(0.1)
                    continue

        idx_buffer.clear()
        return is_all_finished

    # ==================================================================
    # Main loop — with overflow recycling and is_finished control
    # ==================================================================
    cur_idx = 0
    idx_buffer: list[int] = []
    data_buffer: list[list] = []
    cache: dict[int, Any] = {}
    is_finished = False
    is_all_finished = False
    _counter = 1.0 if no_warmup else 0.0

    while True:
        if is_all_finished:
            break

        if done_event.is_set() and not is_finished:
            is_finished = True

        if is_finished:
            if len(idx_buffer) > 0:
                is_all_finished = collate_once(idx_buffer, data_buffer, is_finished=False)
            if not is_all_finished:
                is_all_finished = collate_once(idx_buffer, data_buffer, is_finished=True)
            continue

        # Check cache first (for out-of-order arrivals)
        if cur_idx in cache:
            r = cache.pop(cur_idx)
        else:
            try:
                r = in_queue.get(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if r is None:
                is_finished = True
                continue

        idx, data = r

        if idx > cur_idx:
            cache[idx] = r
            continue
        elif idx < cur_idx:
            raise ValueError(f"Unexpected idx {idx} < cur_idx {cur_idx} in collate loop.")

        if data is None:
            is_finished = True
            continue

        cur_idx += 1

        if not isinstance(data, list):
            out_queue.put((idx, data), timeout=_utils.MP_STATUS_CHECK_INTERVAL)
            continue

        idx_buffer.append(idx)
        data_buffer.append(data)

        # Trigger grouping when buffer is sufficiently full
        if len(idx_buffer) >= int(buffer_size * _counter):
            is_all_finished = collate_once(idx_buffer, data_buffer, is_finished=False)
            # collate_once clears idx_buffer and places overflow back into data_buffer.
            # Do NOT rebind with `= []` — that would discard overflow samples.
            _counter = min(1.0, _counter + 0.05)

    # ==================================================================
    # Cleanup
    # ==================================================================
    has_unexpected_data = (not is_all_finished) and (len(data_buffer) > 0 or len(cache) > 0)

    if ddp_group is not None and not join_mode:
        dist.barrier(ddp_group)
    if ddp_group is not None:
        dist.destroy_process_group(ddp_group)

    idx_buffer.clear()
    data_buffer.clear()
    cache.clear()
    del idx_buffer, data_buffer, cache

    if has_unexpected_data:
        raise RuntimeError("collate_loop exits with remaining data.")
