#!/usr/bin/env bash
# Run manually inside llm200. Baseline always precedes optimized on the same GPU.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
TASK_HOME="$(cd "$PROJECT_DIR/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$TASK_HOME/models/Qwen3-0.6B-Base}"
TRAIN_FILE="${TRAIN_FILE:-$TASK_HOME/datasets/gsm8k/train.jsonl}"
TEST_FILE="${TEST_FILE:-$TASK_HOME/datasets/gsm8k/test.jsonl}"
COMPARE_DIR="${COMPARE_DIR:-$PROJECT_DIR/outputs/full_06b_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
PPO_BATCHES="${PPO_BATCHES:-2491}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1024}"
BUFFER_SIZE="${BUFFER_SIZE:-64}"
SFT_LR="${SFT_LR:-0.00002}"
PPO_LR="${PPO_LR:-0.0000002}"
BASELINE_SFT="${BASELINE_SFT:-}"
SAVE_EVERY="${SAVE_EVERY:-250}"
CONTINUE="${CONTINUE:-0}"
export DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Honor existing CUDA_VISIBLE_DEVICES and optional GPU=index/UUID restrictions.
CUDA_VISIBLE_DEVICES="$(python tools/select_full_gpu.py)"
export CUDA_VISIBLE_DEVICES
if [[ "$CONTINUE" == 1 ]]; then
    test -d "$COMPARE_DIR" || { echo "CONTINUE=1 needs an existing COMPARE_DIR" >&2; exit 1; }
else
    test ! -e "$COMPARE_DIR" || { echo "Choose a new COMPARE_DIR or set CONTINUE=1: $COMPARE_DIR" >&2; exit 1; }
fi
mkdir -p "$COMPARE_DIR"
export HF_HOME="$COMPARE_DIR/hf_cache"
export XDG_CACHE_HOME="$COMPARE_DIR/cache"
# Keep Unix-socket paths short, while still keeping temporary files in ppo.
export TMPDIR="$PROJECT_DIR/.tmp"
mkdir -p "$TMPDIR"
if [[ "$CONTINUE" == 1 ]]; then
    # Keep both experiments on the original GPU when continuing a pair.
    test "$(cat "$COMPARE_DIR/GPU_UUID.txt")" = "$CUDA_VISIBLE_DEVICES" || {
        echo "Selected GPU differs from the original GPU_UUID.txt; specify the original GPU." >&2; exit 1;
    }
    python run_full.py doctor --gpu 2>&1 | tee "$COMPARE_DIR/doctor.continue.log"
    nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv > "$COMPARE_DIR/gpu_continue.csv"
else
    python run_full.py doctor --gpu 2>&1 | tee "$COMPARE_DIR/doctor.log"
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$COMPARE_DIR/GPU_UUID.txt"
    nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv > "$COMPARE_DIR/gpu_before.csv"
fi

for variant in baseline optimized; do
    run_dir="$COMPARE_DIR/$variant"
    if [[ "$CONTINUE" == 1 && -d "$run_dir" ]]; then
        python run_full.py verify --run-dir "$run_dir"
    else
        python run_full.py prepare --profile "$variant" --run-dir "$run_dir" \
            --model "$MODEL_DIR" --train "$TRAIN_FILE" --test "$TEST_FILE" \
            --odb-token-budget "$TOKEN_BUDGET" --odb-buffer-size "$BUFFER_SIZE" \
            --sft-lr "$SFT_LR" --ppo-lr "$PPO_LR" 2>&1 | tee "$COMPARE_DIR/${variant}_prepare.log"
    fi
    if [[ "$variant" == baseline && -n "$BASELINE_SFT" ]]; then
        test -f "$BASELINE_SFT/config.json" || { echo "BASELINE_SFT must be a full HF model directory" >&2; exit 1; }
        echo "Reusing baseline SFT: $BASELINE_SFT"
    elif [[ "$CONTINUE" == 1 && -f "$run_dir/sft/SUCCESS.json" ]]; then
        python - "$run_dir/sft/SUCCESS.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    success = json.load(handle)
assert success['passed'] and success['full_dataset_completed'] and success['examples'] == 7473
PY
        echo "Reusing completed $variant SFT."
    else
        python run_full.py sft --run-dir "$run_dir" 2>&1 | tee "$run_dir/sft.stdout.log"
    fi
    if [[ "$CONTINUE" == 1 && ( -f "$run_dir/runs/full/SUCCESS.json" || -f "$run_dir/runs/full/QUALITY_STOPPED.json" || -f "$run_dir/runs/full/QUALITY_ALERT.json" ) ]]; then
        echo "Reusing completed/stopped $variant PPO."
        continue
    fi
    PPO_SFT_ARG=()
    if [[ "$variant" == baseline && -n "$BASELINE_SFT" ]]; then
        PPO_SFT_ARG=(--sft "$BASELINE_SFT")
    fi
    if python run_full.py ppo --run-dir "$run_dir" --name full "${PPO_SFT_ARG[@]}" \
        --limit "$PPO_BATCHES" --save-every "$SAVE_EVERY" 2>&1 | tee -a "$run_dir/ppo.stdout.log"; then
        :
    elif [[ -f "$run_dir/runs/full/QUALITY_STOPPED.json" ]]; then
        echo "$variant PPO triggered its recorded quality guard; retaining outputs and running the next variant."
    else
        echo "$variant PPO failed; inspect $run_dir/ppo.stdout.log" >&2
        exit 1
    fi
done
python run_full.py verify --run-dir "$COMPARE_DIR/baseline"
python run_full.py verify --run-dir "$COMPARE_DIR/optimized"
python tools/compare_full.py --baseline "$COMPARE_DIR/baseline" --optimized "$COMPARE_DIR/optimized"
echo "Completed. Comparison and all outputs: $COMPARE_DIR"
