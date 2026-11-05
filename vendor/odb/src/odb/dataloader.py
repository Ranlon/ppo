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

"""DataLoader replacement API for Online Dynamic Batching."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .config import ODBConfig
from .core import apply
from .handle import ODBHandle


class ODBDataLoader(DataLoader):
    """A ``DataLoader`` replacement that enables ODB during construction.

    Use this for new PyTorch code where you control DataLoader construction.
    For frameworks that construct the DataLoader internally, use
    :func:`odb.apply` on the existing DataLoader instead.
    """

    odb_handle: ODBHandle

    def __init__(
        self,
        dataset,
        *,
        token_budget: int | None = None,
        config: ODBConfig | None = None,
        max_input_length: int | None = None,
        loss_scaling: bool | str | None = None,
        loss_scaling_approx: bool = True,
        join: bool | None = None,
        join_mode: bool | None = None,
        no_warmup: bool | None = None,
        no_sync: bool | None = None,
        buffer_size: int | None = None,
        version: int | None = None,
        max_patches: int | None = None,
        group_order_flip: str | None = None,
        **dataloader_kwargs: Any,
    ) -> None:
        batch_size = dataloader_kwargs.setdefault("batch_size", 1)
        if batch_size != 1:
            raise ValueError(f"ODBDataLoader requires batch_size=1, got {batch_size}")
        if int(dataloader_kwargs.get("num_workers", 0) or 0) <= 0:
            dataloader_kwargs["num_workers"] = 4
            dataloader_kwargs.setdefault("prefetch_factor", 2)
        super().__init__(dataset, **dataloader_kwargs)
        self.odb_handle = apply(
            self,
            max_input_length=max_input_length,
            token_budget=token_budget,
            config=config,
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
