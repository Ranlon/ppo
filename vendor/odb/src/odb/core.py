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

"""Core ODB entry point: ``apply()`` and the custom DataLoader iterator."""

from __future__ import annotations

import functools
import inspect
import logging
import multiprocessing
import os
import queue
import random
import threading
import time
from types import MethodType
from typing import Any

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import (
    _DatasetKind,
    _MultiProcessingDataLoaderIter,
    _sharding_worker_init_fn,
    _utils,
    IterDataPipe,
    MapDataPipe,
)

from .collate import collate_loop_odb
from .config import ODBConfig, resolve_config
from .constants import IDLE_DATA, LOCAL_BATCH_SIZE_KEY, ODB_STEP_INFO_KEY
from .handle import ODBHandle
from .utils import null_collate_fn

logger = logging.getLogger(__name__)

# PyTorch version compat: _process_data signature
_process_data_sig = inspect.signature(_MultiProcessingDataLoaderIter._process_data)
_PROCESS_DATA_NEEDS_WORKER_IDX = len(_process_data_sig.parameters) > 2

# Set ODB_PROFILE_PATCHES=1 to enable per-batch patch/token diagnostics.
_PROFILE_PATCHES = os.environ.get("ODB_PROFILE_PATCHES", "0") == "1"


def _prefetch_capacity(dataloader: DataLoader) -> int:
    """Return the number of single samples PyTorch can prefetch locally."""
    num_workers = int(getattr(dataloader, "num_workers", 0) or 0)
    if num_workers <= 0:
        return 0
    prefetch_factor = getattr(dataloader, "prefetch_factor", None)
    if prefetch_factor is None:
        prefetch_factor = 2
    return int(prefetch_factor) * num_workers


def _ensure_worker_prefetching(dataloader: DataLoader, *, default_workers: int = 4) -> None:
    """Match the paper LLaMA-Factory path by enabling worker prefetching for ODB."""
    num_workers = int(getattr(dataloader, "num_workers", 0) or 0)
    if num_workers > 0:
        if getattr(dataloader, "prefetch_factor", None) is None:
            dataloader.prefetch_factor = 2
        return

    dataloader.num_workers = int(default_workers)
    if getattr(dataloader, "prefetch_factor", None) is None:
        dataloader.prefetch_factor = 2
    logger.warning(
        "[ODB] DataLoader num_workers=0 requested; enabling worker prefetching "
        "with num_workers=%s and prefetch_factor=%s before iteration.",
        dataloader.num_workers,
        dataloader.prefetch_factor,
    )


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except ValueError:
        return default



def _profile_count_patches(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    pv = data.get("pixel_values")
    if isinstance(pv, torch.Tensor) and pv.dim() >= 1:
        return int(pv.shape[0])
    grid = data.get("image_grid_thw")
    if isinstance(grid, torch.Tensor) and grid.dim() == 2 and grid.shape[1] == 3:
        return int(grid.prod(dim=1).sum().item())
    return 0


def _profile_seq_total(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    ids = data.get("input_ids")
    if isinstance(ids, torch.Tensor):
        return int(ids.numel())
    return 0


def apply(
    dataloader: DataLoader,
    max_input_length: int | None = None,
    loss_scaling: bool | str | None = None,
    loss_scaling_approx: bool = True,
    join_mode: bool | None = None,
    *,
    token_budget: int | None = None,
    config: ODBConfig | None = None,
    join: bool | None = None,
    no_warmup: bool | None = None,
    no_sync: bool | None = None,
    buffer_size: int | None = None,
    version: int | None = None,
    max_patches: int | None = None,
    group_order_flip: str | None = None,
) -> ODBHandle:
    """Apply Online Dynamic Batching to an existing DataLoader **in-place**.

    New code should prefer ``token_budget`` or ``config=ODBConfig(...)``.  The
    legacy ``max_input_length`` and ``join_mode`` names remain as a thin
    compatibility layer for existing callers.
    """
    resolved_config = resolve_config(
        config=config,
        max_input_length=max_input_length,
        token_budget=token_budget,
        loss_scaling=loss_scaling,
        loss_scaling_approx=loss_scaling_approx,
        join=join,
        join_mode=join_mode,
        no_warmup=no_warmup,
        no_sync=no_sync,
        buffer_size=buffer_size,
        version=version,
        max_patches=max_patches,
        group_order_flip=group_order_flip,
    )
    handle = ODBHandle(config=resolved_config, step_info_key=ODB_STEP_INFO_KEY)

    from .grouping import normalize_group_order_flip

    resolved_group_order_flip = normalize_group_order_flip(resolved_config.group_order_flip)
    token_budget = resolved_config.token_budget
    loss_scaling_enabled = resolved_config.loss_scaling_enabled
    loss_scaling_approx_enabled = resolved_config.loss_scaling_approx
    join_enabled = resolved_config.join
    no_warmup_enabled = resolved_config.no_warmup
    no_sync_enabled = resolved_config.no_sync
    resolved_buffer_size = resolved_config.buffer_size
    resolved_max_patches = resolved_config.max_patches

    def _distributed_sync_enabled() -> bool:
        return (not no_sync_enabled) and dist.is_available() and dist.is_initialized()

    if resolved_config.version == 9:
        if resolved_group_order_flip != "none":
            raise ValueError("group_order_flip is only supported by ODB v5.1, not version=9.")
        try:
            from .core_v9 import apply_v9
        except ImportError as exc:
            raise ValueError("ODB version=9 is not included in this standalone package build.") from exc

        apply_v9(
            dataloader,
            token_budget,
            loss_scaling=loss_scaling_enabled,
            loss_scaling_approx=loss_scaling_approx_enabled,
            no_warmup=no_warmup_enabled,
            no_sync=no_sync_enabled,
            buffer_size=resolved_buffer_size,
        )
        dataloader._odb_config = resolved_config
        dataloader._odb_handle = handle
        return handle

    if dataloader.batch_size != 1:
        raise ValueError(
            f"ODB requires DataLoader with batch_size=1, got {dataloader.batch_size}. "
            f"ODB dynamically determines batch size via grouping."
        )
    _ensure_worker_prefetching(dataloader)

    if resolved_buffer_size is not None and resolved_buffer_size < 1:
        raise ValueError(f"buffer_size must be >= 1, got {resolved_buffer_size}")

    if resolved_buffer_size is not None and not no_sync_enabled and dist.is_initialized():
        dataset_len = len(dataloader.dataset) if hasattr(dataloader.dataset, "__len__") else None
        if dataset_len is not None:
            est_grouping_rounds = dataset_len // max(resolved_buffer_size, 1)
            default_buf = _prefetch_capacity(dataloader)
            est_normal_rounds = dataset_len // default_buf if default_buf > 0 else 0

            if est_grouping_rounds > 10 * est_normal_rounds:
                import warnings

                warnings.warn(
                    f"[ODB] buffer_size={resolved_buffer_size} will trigger ~{est_grouping_rounds} "
                    f"Gloo all_gather calls (vs ~{est_normal_rounds} with default buffer_size={default_buf}). "
                    "High-frequency all_gather may exhaust Gloo TCP resources and cause SIGABRT. "
                    "If training crashes, increase buffer_size or set no_sync=True.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _get_iterator(self):
        assert self.num_workers > 0, "ODB requires num_workers > 0"
        if dist.is_initialized():
            jitter_s = _env_float("ODB_ITERATOR_STARTUP_JITTER_SEC", 2.0)
            if jitter_s > 0:
                time.sleep(jitter_s * random.random())
        self.check_worker_number_rationality()

        if not hasattr(self, "_iter_random_seed"):
            self._iter_random_seed = 67108
        else:
            self._iter_random_seed += 1

        effective_join_enabled = join_enabled and _distributed_sync_enabled()
        self._odb_effective_join_mode = effective_join_enabled
        iterator = _OnlineDynamicBatchIter(
            self,
            token_budget,
            self._iter_random_seed,
            loss_scaling_enabled,
            loss_scaling_approx_enabled,
            effective_join_enabled,
            no_warmup_enabled,
            no_sync_enabled,
            resolved_buffer_size,
            resolved_max_patches,
            resolved_group_order_flip,
        )
        self.__dict__.setdefault("_odb_active_iterators", []).append(iterator)
        return iterator

    if hasattr(dataloader, "_get_iterator"):
        dataloader._get_iterator = MethodType(_get_iterator, dataloader)
    else:
        raise AttributeError(f"{type(dataloader)} has no '_get_iterator' attribute, which is required for ODB.")

    dataloader._odb_enabled = True
    dataloader._odb_max_input_length = token_budget
    dataloader._odb_token_budget = token_budget
    dataloader._odb_loss_scaling = loss_scaling_enabled
    dataloader._odb_loss_scaling_mode = resolved_config.loss_scaling
    dataloader._odb_join_mode = join_enabled
    dataloader._odb_effective_join_mode = join_enabled and _distributed_sync_enabled()
    dataloader._odb_group_order_flip = resolved_group_order_flip
    dataloader._odb_config = resolved_config
    dataloader._odb_handle = handle
    return handle


def close_odb_loader(dataloader: DataLoader) -> None:
    """Release ODB workers after a caller stops before dataset exhaustion.

    In DDP join mode the iterator deliberately remains alive after early stop.
    Call this after all ranks have finished consuming training batches.
    """
    iterators = getattr(dataloader, "_odb_active_iterators", None)
    if not iterators:
        return
    for iterator in iterators:
        if not iterator._shutdown:
            iterator._join_mode = False
            iterator._shutdown_workers()
    iterators.clear()


# ======================================================================
# Iterator
# ======================================================================


class _OnlineDynamicBatchIter(_MultiProcessingDataLoaderIter):
    """Custom DataLoader iterator that performs ODB grouping in a collate process."""

    def __init__(
        self,
        loader: DataLoader,
        max_input_length: int,
        random_seed: int,
        loss_scaling: bool = False,
        loss_scaling_approx: bool = True,
        join_mode: bool = False,
        no_warmup: bool = False,
        no_sync: bool = False,
        buffer_size_override: int | None = None,
        max_patches: int = 0,
        group_order_flip: str | None = "none",
    ):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._dataset_kind == _DatasetKind.Map, "ODB only supports Map-style datasets"
        assert not self._persistent_workers, "ODB doesn't support persistent workers"

        self._prefetch_factor = loader.prefetch_factor if loader.prefetch_factor is not None else 2
        self._join_mode = join_mode
        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            mp_context = multiprocessing
        else:
            mp_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn,
                self._worker_init_fn,
                self._world_size,
                self._rank,
            )

        # Optional dataset wrapper for identity-coverage logging.
        # Activated by env var ODB_LOG_EMITTED_IDS (path template); the
        # wrapper injects "_odb_sample_idx" into each item dict so that
        # collate.py can log it. No-op when env var is unset.
        if os.environ.get("ODB_LOG_EMITTED_IDS"):
            from .utils import _IDInjectingDataset
            self._dataset = _IDInjectingDataset(self._dataset)
            logger.info("[ODB][rank %s] _IDInjectingDataset enabled", self._rank)

        # Queues
        self._worker_result_queue = mp_context.Queue()
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = mp_context.Event()

        # Worker processes
        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = mp_context.Queue()
            index_queue.cancel_join_thread()
            w = mp_context.Process(
                target=_utils.worker._worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    null_collate_fn,
                    self._drop_last,
                    self._base_seed,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                    self._shared_seed,
                ),
            )
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        # Collate process
        self._collate_queue = mp_context.Queue()
        if buffer_size_override is not None:
            buffer_size = buffer_size_override
        else:
            buffer_size = self._prefetch_factor * self._num_workers
        self._odb_buffer_size = buffer_size
        logger.info("[ODB] buffer_size=%s (override=%s)", buffer_size, buffer_size_override)

        collate_process = mp_context.Process(
            target=collate_loop_odb,
            args=(
                self._worker_result_queue,
                self._collate_queue,
                self._collate_fn,
                buffer_size,
                max_input_length,
                self._workers_done_event,
                random_seed,
                loss_scaling,
                loss_scaling_approx,
                join_mode,
                no_warmup,
                no_sync,
                max_patches,
                group_order_flip,
            ),
        )
        collate_process.daemon = True
        collate_process.start()
        self._collate_process = collate_process

        # Pin-memory thread
        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._data_queue = queue.Queue()

            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()
            elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
                custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
                current_device = custom_device_mod.current_device()
            else:
                current_device = torch.cuda.current_device()

            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(
                    self._collate_queue,
                    self._data_queue,
                    current_device,
                    self._pin_memory_thread_done_event,
                    self._pin_memory_device,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._collate_queue

        if self._persistent_workers and self._pin_memory:
            import atexit

            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        _utils.signal_handling._set_worker_pids(
            id(self),
            tuple(w.pid for w in self._workers) + (self._collate_process.pid,),
        )
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _profile_patches_emit(self, data: Any) -> None:
        """Emit a patch-profile line per yielded batch when enabled.

        Writes to the package logger and, when possible, a per-rank file under
        ``/tmp`` for offline diagnostics.
        """
        if not _PROFILE_PATCHES or not isinstance(data, dict):
            return
        if not hasattr(self, "_profile_step_counter"):
            self._profile_step_counter = 0
            self._profile_rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
            self._profile_log_path = f"/tmp/odb_patch_prof_rank{self._profile_rank}.log"
            try:
                self._profile_log_fp = open(self._profile_log_path, "w")
            except OSError:
                self._profile_log_fp = None
        self._profile_step_counter += 1
        local_bs = int(data.get(LOCAL_BATCH_SIZE_KEY, 0))
        seq_tot = _profile_seq_total(data)
        n_patch = _profile_count_patches(data)
        line = (
            f"[PATCH-PROF rank{self._profile_rank}] step={self._profile_step_counter} "
            f"bs={local_bs} seq_total={seq_tot} n_patches={n_patch}"
        )
        logger.info(line)
        if self._profile_log_fp is not None:
            self._profile_log_fp.write(line + "\n")
            self._profile_log_fp.flush()

    def _reset(self, loader, first_iter=False):
        self._collate_status = True
        self._end_of_data = False
        super()._reset(loader, first_iter)
        # When buffer_size > prefetch_factor * num_workers, the default prefetch
        # is insufficient for the first grouping round.  Send extra indices so the
        # collate subprocess can fill its buffer without deadlocking.
        default_prefetch = self._prefetch_factor * self._num_workers
        if self._odb_buffer_size > default_prefetch:
            extra = self._odb_buffer_size - default_prefetch
            for _ in range(extra):
                self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            if self._collate_status and not self._collate_process.is_alive():
                raise RuntimeError(
                    f"DataLoader collate process (pid {self._collate_process.pid}) exited unexpectedly"
                )

            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)

            if len(failed_workers) > 0:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly") from e

            if isinstance(e, queue.Empty):
                return (False, None)

            import errno
            import tempfile

            try:
                fds_limit_margin = 10
                _ = [tempfile.NamedTemporaryFile() for _ in range(fds_limit_margin)]
            except OSError as e2:
                if e2.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the workers is no longer possible. "
                        "Please increase the limit using `ulimit -n` in the shell or change the "
                        "sharing strategy by calling "
                        "`torch.multiprocessing.set_sharing_strategy('file_system')` "
                        "at the beginning of your code"
                    ) from None
            raise

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info.get(self._rcvd_idx, None)
                if info:
                    worker_id = info[0]
                    if len(info) == 2 or self._workers_status[worker_id]:
                        break
                    del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                if not self._persistent_workers and not self._join_mode:
                    self._shutdown_workers()
                raise StopIteration

            if len(self._task_info[self._rcvd_idx]) == 2:
                worker_id, data = self._task_info.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                if _PROCESS_DATA_NEEDS_WORKER_IDX:
                    data = self._process_data(data, worker_id)
                else:
                    data = self._process_data(data)
                if data == IDLE_DATA:
                    continue
                if isinstance(data, dict) and (not data or "input_ids" not in data):
                    continue
                self._profile_patches_emit(data)
                return data

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1

            if self._dataset_kind == _DatasetKind.Iterable:
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                self._task_info[idx] += (data,)
            else:
                worker_id = self._task_info.pop(idx)[0]
                self._rcvd_idx += 1
                if _PROCESS_DATA_NEEDS_WORKER_IDX:
                    data = self._process_data(data, worker_id)
                else:
                    data = self._process_data(data)
                if data == IDLE_DATA:
                    continue
                if isinstance(data, dict) and (not data or "input_ids" not in data):
                    continue
                self._profile_patches_emit(data)
                return data

    def _try_put_index(self):
        max_outstanding = max(self._prefetch_factor * self._num_workers, self._odb_buffer_size)
        if self._tasks_outstanding >= max_outstanding:
            return

        try:
            index = self._next_index()
        except StopIteration:
            if not self._end_of_data:
                self._worker_result_queue.put((self._send_idx, None))
                self._end_of_data = True
            return

        for _ in range(self._num_workers):
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _mark_collate_as_unavailable(self, shutdown=False):
        assert self._collate_status or (self._persistent_workers and shutdown)
        self._collate_queue.put(None)
        self._collate_status = False
        assert self._workers_done_event.is_set() == shutdown

    def __del__(self):
        """Best-effort cleanup.

        In join mode, StopIteration can happen before the collate subprocess is
        done because that subprocess must keep serving ODB's Gloo group while
        other ranks finish.  The normal shutdown path is therefore delayed until
        the subprocess has exited.
        """
        if getattr(self, "_shutdown", True):
            return

        if getattr(self, "_join_mode", False):
            collate_process = getattr(self, "_collate_process", None)
            if collate_process is not None and not collate_process.is_alive():
                self._shutdown_workers()
        else:
            self._shutdown_workers()

    def _shutdown_workers(self):
        if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
            return

        if not self._shutdown:
            self._shutdown = True
            try:
                if hasattr(self, "_pin_memory_thread"):
                    self._pin_memory_thread_done_event.set()
                    self._collate_queue.put((None, None))
                    self._pin_memory_thread.join()

                self._worker_result_queue.put((self._send_idx, None))
                self._worker_result_queue.cancel_join_thread()
                self._worker_result_queue.close()

                self._workers_done_event.set()
                self._mark_collate_as_unavailable(shutdown=True)
                for worker_id in range(len(self._workers)):
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)

                _JOIN_TIMEOUT = 30 if self._join_mode else _utils.MP_STATUS_CHECK_INTERVAL
                self._collate_process.join(timeout=_JOIN_TIMEOUT)
                for w in self._workers:
                    w.join(timeout=_JOIN_TIMEOUT)

                if hasattr(self, "_pin_memory_thread"):
                    self._collate_queue.cancel_join_thread()
                    self._collate_queue.close()

                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()

            finally:
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False

                alive_workers = [w for w in self._workers if w.is_alive()]
                if self._collate_process.is_alive() or alive_workers:
                    for q in self._index_queues:
                        try:
                            q.put_nowait(None)
                        except Exception:
                            pass

                    _GRACE_TIMEOUT = 5
                    if self._collate_process.is_alive():
                        self._collate_process.join(timeout=_GRACE_TIMEOUT)
                    for w in alive_workers:
                        w.join(timeout=_GRACE_TIMEOUT)

                    if self._collate_process.is_alive():
                        self._collate_process.terminate()
                    for w in self._workers:
                        if w.is_alive():
                            w.terminate()
                    self._collate_process.join(timeout=1)
                    for w in self._workers:
                        w.join(timeout=1)
