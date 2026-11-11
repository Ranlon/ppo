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

"""HuggingFace Trainer integration for ODB.

Requires the ``transformers`` package (install with ``pip install online-dynamic-batching[hf]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import Trainer

from .step_info import ODBStepInfo, pop_step_info

try:
    from transformers.trainer_callback import TrainerCallback

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

    class TrainerCallback:  # type: ignore[no-redef]
        """Stub when transformers is not installed."""


class ODBDynamicBatchCallback(TrainerCallback):
    """Callback for correct epoch tracking with ODB dynamic batching.

    Standard HuggingFace Trainer calculates epoch as ``step / total_steps``,
    but with ODB each step processes a variable number of samples.  This
    callback tracks the actual number of samples processed and computes epoch
    correctly.  It also compensates the LR scheduler so that the cosine (or
    other) schedule is aligned to sample progress rather than step progress.

    Args:
        odb_total_samples: Total number of samples to process across all
            epochs and ranks (``len(dataloader) * num_epochs * world_size``).
    """

    def __init__(
        self,
        odb_total_samples: int | None = None,
        *,
        sample_budget: int | None = None,
        scheduler_progress: str = "samples",
        max_optimizer_steps: int | None = None,
    ):
        if not _HF_AVAILABLE:
            raise ImportError(
                "ODBDynamicBatchCallback requires the `transformers` package. "
                "Install with: pip install online-dynamic-batching[hf]"
            )
        if sample_budget is None:
            sample_budget = odb_total_samples
        if sample_budget is None:
            raise ValueError("ODBDynamicBatchCallback requires sample_budget or legacy odb_total_samples")
        if scheduler_progress not in {"samples", "optimizer_steps"}:
            raise ValueError("scheduler_progress must be 'samples' or 'optimizer_steps'")
        self._lr_scheduler_last_epoch = -10000
        self._odb_total_samples = int(sample_budget)
        self._scheduler_progress = scheduler_progress
        self._max_optimizer_steps = max_optimizer_steps

    def on_train_begin(self, args, state, control, **kwargs):
        state.total_data_step = 0
        state.total_batch_sizes = []
        state.odb_step_infos = []
        state.accumulation_batch_size = 0
        state.odb_optimizer_steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._update_step(args, state, control)
        assert len(state.total_batch_sizes) == 0, (
            f"total_batch_sizes should be empty after step_end, got {len(state.total_batch_sizes)} items"
        )
        assert len(getattr(state, "odb_step_infos", [])) == 0, (
            f"odb_step_infos should be empty after step_end, got {len(state.odb_step_infos)} items"
        )

        lr_scheduler = kwargs.get("lr_scheduler")
        if (
            self._scheduler_progress == "samples"
            and lr_scheduler is not None
            and self._lr_scheduler_last_epoch < lr_scheduler.last_epoch
        ):
            for _ in range(state.accumulation_batch_size - 1):
                lr_scheduler.step()
            self._lr_scheduler_last_epoch = lr_scheduler.last_epoch
        state.accumulation_batch_size = 0
        state.odb_optimizer_steps += 1
        if self._max_optimizer_steps is not None and state.odb_optimizer_steps >= self._max_optimizer_steps:
            control.should_training_stop = True

    def on_substep_end(self, args, state, control, **kwargs):
        self._update_step(args, state, control)

    def _update_step(self, args, state, control):
        step_info = None
        if getattr(state, "odb_step_infos", None):
            step_info = state.odb_step_infos.pop(0)
        elif state.total_batch_sizes:
            step_info = ODBStepInfo(all_samples_this_step=int(state.total_batch_sizes.pop(0)), loss_scale=1.0)
        else:
            return

        all_samples_this_step = int(step_info.all_samples_this_step)
        state.accumulation_batch_size += all_samples_this_step
        state.total_data_step += all_samples_this_step
        state.epoch = state.total_data_step / self._odb_total_samples * args.num_train_epochs

        if state.total_data_step >= self._odb_total_samples:
            control.should_training_stop = True


def setup_odb_training(
    trainer: "Trainer",
    dataloader: "DataLoader",
    max_input_length: int,
    loss_scaling: bool = False,
    loss_scaling_approx: bool = True,
    join_mode: bool = True,
) -> None:
    """One-call setup: apply ODB to a dataloader and register the trainer callback.

    This is the recommended way to integrate ODB with HuggingFace Trainer.
    It performs all necessary steps:

    1. Apply ODB to the dataloader (``odb.apply``).
    2. Calculate the total sample target.
    3. Set ``args.max_steps`` for the LR scheduler.
    4. Disable ``dataloader.__len__`` to prevent Trainer from overriding max_steps.
    5. Register :class:`ODBDynamicBatchCallback`.
    6. Wrap ``trainer.compute_loss`` to auto-apply loss scaling.

    Args:
        trainer: A HuggingFace ``Trainer`` instance.
        dataloader: The training ``DataLoader`` (before ODB is applied).
        max_input_length: Reference length for batch-size calculation.
        loss_scaling: Enable token-level loss scaling for DDP gradient correction.
        loss_scaling_approx: Use approximate mode (default) or exact mode.
        join_mode: Forwarded to :func:`odb.apply`; defaults to ``True``.
            Pair this with DDP Join / ``Accelerator.join_uneven_inputs`` if
            ranks may finish at different optimizer steps.

    Example::

        from odb.callbacks import setup_odb_training

        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        dataloader = trainer.get_train_dataloader()
        setup_odb_training(trainer, dataloader, max_input_length=16384, loss_scaling=True)
    """
    from .integrations.hf import configure_trainer

    sample_budget = int(len(dataloader.dataset) * trainer.args.num_train_epochs)
    return configure_trainer(
        trainer,
        dataloader=dataloader,
        max_input_length=max_input_length,
        sample_budget=sample_budget,
        loss_scaling=loss_scaling,
        loss_scaling_approx=loss_scaling_approx,
        join_mode=join_mode,
        max_steps_policy="overwrite",
    )


def _wrap_compute_loss(trainer: "Trainer", *, loss_scaling: bool | str | None = "auto") -> None:
    """Wrap trainer.compute_loss to auto-apply ODB loss scaling.

    Intercepts the ``inputs`` dict to:
    1. Extract ``total_batch_size`` for the epoch callback.
    2. Apply token-level loss scaling for correct DDP gradient averaging.
    3. Track unscaled loss running average for logging.
    """
    original_compute_loss = trainer.compute_loss

    # Running average state for unscaled loss
    _unscaled_loss_sum = [0.0]
    _unscaled_loss_count = [0]

    def _odb_compute_loss(model, inputs, *args, **kwargs):
        step_info = pop_step_info(inputs, loss_scaling=loss_scaling)
        if hasattr(trainer.state, "odb_step_infos"):
            trainer.state.odb_step_infos.append(step_info)
        elif hasattr(trainer.state, "total_batch_sizes"):
            trainer.state.total_batch_sizes.append(step_info.all_samples_this_step)

        # Call original compute_loss
        loss = original_compute_loss(model, inputs, *args, **kwargs)

        scale = step_info.loss_scale
        should_scale = not isinstance(scale, (int, float)) or float(scale) != 1.0
        if should_scale:
            target_loss = loss[0] if isinstance(loss, tuple) else loss
            _unscaled_loss_sum[0] += target_loss.detach().item()
            _unscaled_loss_count[0] += 1
            scaled_loss = target_loss * scale
            if isinstance(loss, tuple):
                loss = (scaled_loss, *loss[1:])
            else:
                loss = scaled_loss

        return loss

    # Attach unscaled loss state for external logging
    _odb_compute_loss._unscaled_loss_sum = _unscaled_loss_sum
    _odb_compute_loss._unscaled_loss_count = _unscaled_loss_count

    trainer.compute_loss = _odb_compute_loss
