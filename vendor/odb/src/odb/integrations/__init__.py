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

"""Framework integrations for ODB."""

from .accelerate import ODBAccelerateBridge, configure_accelerator
from .hf import ODBTrainer, ODBTrainerBridge, ODBTrainerMixin, configure_trainer as configure_hf_trainer
from .hf import enable_odb as enable_hf_odb
from .lightning import (
    ODBLightningBridge,
    ODBLightningCallback,
    ODBLightningMixin,
    configure_lightning_module,
)
from .llamafactory import configure_trainer as configure_llamafactory_trainer
from .llamafactory import enable_odb as enable_llamafactory_odb

configure_trainer = configure_hf_trainer

__all__ = [
    "ODBAccelerateBridge",
    "ODBLightningBridge",
    "ODBLightningCallback",
    "ODBLightningMixin",
    "ODBTrainer",
    "ODBTrainerBridge",
    "ODBTrainerMixin",
    "configure_accelerator",
    "configure_lightning_module",
    "configure_trainer",
    "configure_hf_trainer",
    "configure_llamafactory_trainer",
    "enable_hf_odb",
    "enable_llamafactory_odb",
]
