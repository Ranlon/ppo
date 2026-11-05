# ODB during the PPO project's SFT stage

Branch: `obd_sft`. The existing `main` commit records the original PPO project.
The legacy PPO environment (TRL 0.9.6, LLaMA-Factory 0.9.4+ppo4.portable1,
PyTorch 2.6/cu124) is used for both stages. No dependency upgrade is needed.

## Start SFT

Run inside the existing llm200 container with the home directory mounted at
`/hpc_stor01/home/mengqi.li_sx`:

```bash
cd /hpc_stor01/home/mengqi.li_sx/Project200/ppo
bash tools/run_odb_sft.sh
```

This chooses an available BF16-capable GPU with at least 16 GiB free, prepares
the local GSM8K `main` Parquet files (7473 train / 1319 test examples), and
trains Qwen3-4B-Base for one full SFT sample epoch. It never starts PPO.
If no suitable GPU is free, the command exits before preparing the run.
Set `GPU=0` (physical nvidia-smi index) or a GPU UUID to restrict the choice.
An inherited `CUDA_VISIBLE_DEVICES` restriction is also respected.

```bash
GPU=0 bash tools/run_odb_sft.sh
```

The launcher uses `run.py` from the mounted checkout; the older copy in
`/opt/ppo` inside the image does not include this branch. The run directory is
printed and defaults to `/hpc_stor01/home/mengqi.li_sx/outputs/ppo/obd_sft_odb_TIMESTAMP`.
Use a fresh `RUN_DIR` to set the output path explicitly. The final adapter is
at `sft/model/`; `sft/SUCCESS.json` records sample coverage, actual input and
supervised token counts, optimizer updates, throughput, batch histogram,
padding fraction, and peak allocated VRAM. `sft.stdout.log` contains the log.

## Settings and comparison

ODB is enabled only by `prepare --sft-odb`. It defaults to a 1024-token budget,
64-sample window, and four DataLoader workers with prefetch 16. Gradient
checkpointing stays enabled with the baseline profile. `TOKEN_BUDGET` and
`BUFFER_SIZE` can be passed to the launcher for separate fresh runs.

The initial implementation supports one device and one optimizer update per
dynamic batch (gradient accumulation 1). Accumulation larger than 1 is rejected:
HF Trainer cannot infer the true dynamic loader length, so a partial final
accumulation window can otherwise cross epoch boundaries. Native SFT remains
batch 1 / accumulation 8. Effective batches and optimizer updates therefore
differ; this branch does not promise equivalent optimization trajectories.

Loss is normalized by the shifted supervised-token count, not input-token
count. Learning rate follows consumed samples with warmup ratio 0.03 and a
cosine schedule. Full completion requires exactly-once coverage of all 7473
samples plus matching input and supervised token totals; 935 updates is only
the native baseline assertion. ODB checkpoint resume is not implemented and
is rejected. The full run has an upper guard of 7473 updates; progress by
samples and `SUCCESS.json` determine completion.

To obtain a fresh native SFT comparison on the same hardware:

```bash
SFT_MODE=baseline bash tools/run_odb_sft.sh
```

Compare equal data coverage, input/supervised tokens per second, peak VRAM,
and held-out quality. Do not attribute effects of changed batch size and
optimizer count solely to length grouping. Original PPO runtime modules,
RL profiles, reward contract, and frozen SFT reference implementation are
unchanged. Neither run loads UltraChat or an Instruct model.

## Continue to PPO

The existing SFT command attaches the completed adapter to the prepared run
only after validation succeeds. Continue with the original PPO baseline using
the printed run directory:

```bash
python run.py ppo --run-dir /path/to/printed/run-directory --name full
```

The reference and actor start from that SFT adapter. For a comparison of PPO
runtime improvements, both PPO variants must use the same SFT adapter.

## Source and verification

ODB is vendored under `vendor/odb`, with its Apache-2.0 license and provenance.
Prepared runs hash that source together with the PPO runtime and LLaMA-Factory
source. The vendored ODB includes the pin-memory shutdown wake-up correction.

CPU integration tests exercise real HF Trainer with a tiny Qwen3 model:
sample coverage including the tail, dynamic batch emission, supervised-token
gradient agreement, and sample-based scheduler boundaries. They do not prove
full-model GPU memory fit or SFT/PPO quality; the formal GPU run is performed
by the user.
