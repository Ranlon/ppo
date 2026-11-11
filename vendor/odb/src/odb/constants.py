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

"""Constants used across the ODB package."""

# Sentinel value for idle/empty data slots — returned when a rank has fewer
# groups than the synchronized target count.
IDLE_DATA = -4201

# Default key name used to inject total_batch_size into each batch dict.
TOTAL_BATCH_SIZE_KEY = "total_batch_size"

# Key name for local batch size (number of samples on this rank).
LOCAL_BATCH_SIZE_KEY = "local_batch_size"

# Key names for token-level loss scaling (injected when loss_scaling=True).
TOTAL_TOKENS_KEY = "odb_total_tokens"
LOCAL_TOKENS_KEY = "odb_local_tokens"

# Reserved key for the clean ODBStepInfo transport. Batches may still use the
# legacy flat keys above; ``odb.pop_step_info`` supports both.
ODB_STEP_INFO_KEY = "odb_step_info"
