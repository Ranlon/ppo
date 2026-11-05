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

"""Online Dynamic Batching (ODB) — PyTorch DataLoader-side dynamic batching.

Quick start::

    import odb
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=1, num_workers=4, prefetch_factor=64)
    odb.apply(dl, token_budget=16384)

    for batch in dl:
        info = odb.pop_step_info(batch)
        ...
"""

from odb._version import __version__
from odb.config import ODBConfig
from odb.constants import (
    LOCAL_BATCH_SIZE_KEY,
    LOCAL_TOKENS_KEY,
    ODB_STEP_INFO_KEY,
    TOTAL_BATCH_SIZE_KEY,
    TOTAL_TOKENS_KEY,
)
from odb.core import apply
from odb.dataloader import ODBDataLoader
from odb.handle import ODBHandle
try:
    from odb.core_v9 import apply_v9
except ImportError:  # optional in minimal standalone package builds
    apply_v9 = None
from odb.step_info import ODBStepInfo, pop_step_info
from odb.utils import scale_loss

__all__ = [
    "__version__",
    "ODBConfig",
    "ODBDataLoader",
    "ODBHandle",
    "ODBStepInfo",
    "apply",
    "apply_v9",
    "pop_step_info",
    "scale_loss",
    "LOCAL_BATCH_SIZE_KEY",
    "LOCAL_TOKENS_KEY",
    "ODB_STEP_INFO_KEY",
    "TOTAL_BATCH_SIZE_KEY",
    "TOTAL_TOKENS_KEY",
]
