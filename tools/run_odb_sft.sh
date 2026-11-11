#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_MODE="${SFT_MODE:-odb}"
case "$SFT_MODE" in baseline|odb) ;; *) echo 'SFT_MODE must be baseline or odb' >&2; exit 2 ;; esac
RUN_DIR="${RUN_DIR:-/hpc_stor01/home/mengqi.li_sx/outputs/ppo/obd_sft_${SFT_MODE}_$(date +%Y%m%d_%H%M%S)}"
MODEL_DIR="${MODEL_DIR:-/hpc_stor01/home/mengqi.li_sx/models/Qwen3-4B-Base}"
DATA_DIR="${DATA_DIR:-/hpc_stor01/home/mengqi.li_sx/datasets/gsm8k}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1024}"
BUFFER_SIZE="${BUFFER_SIZE:-64}"

cd "$PROJECT_DIR"
export DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES="$(python tools/select_sft_gpu.py)"
export CUDA_VISIBLE_DEVICES
python run.py doctor --gpu

EXTRA_ARGS=()
if [[ "$SFT_MODE" == odb ]]; then
    EXTRA_ARGS=(--sft-odb --odb-token-budget "$TOKEN_BUDGET" --odb-buffer-size "$BUFFER_SIZE")
fi
echo "SFT mode: $SFT_MODE; run directory: $RUN_DIR"
python run.py prepare --profile baseline --run-dir "$RUN_DIR" \
    --model "$MODEL_DIR" \
    --train "$DATA_DIR/main/train-00000-of-00001.parquet" \
    --test "$DATA_DIR/main/test-00000-of-00001.parquet" \
    "${EXTRA_ARGS[@]}"

python run.py sft --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/sft.stdout.log"
echo "SFT completed. Result: $RUN_DIR/sft/SUCCESS.json"
echo "PPO-compatible SFT adapter: $RUN_DIR/sft/model"
