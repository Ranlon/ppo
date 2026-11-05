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

"""Public configuration objects for Online Dynamic Batching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


LossScalingMode = Literal["none", "approx", "exact"]
GroupOrderFlip = Literal[
    "none",
    "rank_epoch_random",
    "rank_window_random",
    "rank_window_balanced",
]


@dataclass(frozen=True)
class ODBConfig:
    """Resolved ODB dataloader configuration.

    ``token_budget`` is the preferred public name for the historical
    ``max_input_length`` argument.
    """

    token_budget: int
    loss_scaling: LossScalingMode = "none"
    join: bool = True
    buffer_size: int | None = None
    max_patches: int = 0
    group_order_flip: GroupOrderFlip | str | None = "none"
    no_sync: bool = False
    no_warmup: bool = False
    version: int = 5

    @property
    def max_input_length(self) -> int:
        """Legacy alias for ``token_budget``."""
        return self.token_budget

    @property
    def join_mode(self) -> bool:
        """Legacy alias for ``join``."""
        return self.join

    @property
    def loss_scaling_enabled(self) -> bool:
        return self.loss_scaling != "none"

    @property
    def loss_scaling_approx(self) -> bool:
        return self.loss_scaling != "exact"


def normalize_loss_scaling(
    loss_scaling: bool | str | None,
    loss_scaling_approx: bool = True,
) -> LossScalingMode:
    """Normalize legacy booleans and clean string modes."""
    if loss_scaling is None or loss_scaling is False:
        return "none"
    if loss_scaling is True:
        return "approx" if loss_scaling_approx else "exact"

    mode = str(loss_scaling).lower().replace("-", "_")
    aliases = {
        "no": "none",
        "off": "none",
        "false": "none",
        "token": "approx",
        "token_approx": "approx",
        "approximate": "approx",
        "token_exact": "exact",
        "true": "approx" if loss_scaling_approx else "exact",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "approx", "exact"}:
        raise ValueError(
            "loss_scaling must be one of 'none', 'approx', or 'exact' "
            f"(or a legacy boolean), got {loss_scaling!r}"
        )
    return mode  # type: ignore[return-value]


def _check_conflict(name: str, configured: object, explicit: object) -> None:
    if explicit is not None and explicit != configured:
        raise ValueError(f"Conflicting ODB {name}: config has {configured!r}, explicit value is {explicit!r}")


def resolve_config(
    *,
    config: ODBConfig | None = None,
    max_input_length: int | None = None,
    token_budget: int | None = None,
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
) -> ODBConfig:
    """Resolve clean and legacy apply arguments into one ``ODBConfig``."""
    if max_input_length is not None and token_budget is not None and max_input_length != token_budget:
        raise ValueError(
            "Conflicting ODB token budget: max_input_length="
            f"{max_input_length!r}, token_budget={token_budget!r}"
        )

    resolved_budget = token_budget if token_budget is not None else max_input_length
    resolved_join = join if join is not None else join_mode

    if config is not None:
        _check_conflict("token_budget", config.token_budget, resolved_budget)
        _check_conflict("join", config.join, resolved_join)
        _check_conflict("buffer_size", config.buffer_size, buffer_size)
        _check_conflict("max_patches", config.max_patches, max_patches)
        _check_conflict("no_sync", config.no_sync, no_sync)
        _check_conflict("no_warmup", config.no_warmup, no_warmup)
        _check_conflict("version", config.version, version)

        if loss_scaling is not None:
            explicit_loss_scaling = normalize_loss_scaling(loss_scaling, loss_scaling_approx)
            _check_conflict("loss_scaling", config.loss_scaling, explicit_loss_scaling)
        if group_order_flip is not None:
            _check_conflict("group_order_flip", config.group_order_flip, group_order_flip)
        return config

    if resolved_budget is None:
        raise TypeError("ODB requires a token budget: pass token_budget=... or legacy max_input_length=...")
    if resolved_budget <= 0:
        raise ValueError(f"token_budget must be > 0, got {resolved_budget}")

    return ODBConfig(
        token_budget=int(resolved_budget),
        loss_scaling=normalize_loss_scaling(loss_scaling, loss_scaling_approx),
        join=bool(resolved_join) if resolved_join is not None else True,
        buffer_size=buffer_size,
        max_patches=0 if max_patches is None else int(max_patches),
        group_order_flip="none" if group_order_flip is None else group_order_flip,
        no_sync=bool(no_sync) if no_sync is not None else False,
        no_warmup=bool(no_warmup) if no_warmup is not None else False,
        version=5 if version is None else int(version),
    )


def replace_config(config: ODBConfig, **changes: object) -> ODBConfig:
    """Small typed wrapper around ``dataclasses.replace``."""
    return replace(config, **changes)
