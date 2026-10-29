#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ckpt_setting is the run directory name; pass it verbatim as ckpt_name to eval.sh.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
# Nest task-finetune runs under checkpoints/<task>/ (e.g. stack_blocks_ckpt0 →
# checkpoints/stack_blocks/<run_dir>); runs without the _ckpt<tag> suffix
# (e.g. base "sim" training) keep the legacy flat layout.
case "${ckpt_name}" in
  *_ckpt*) ckpt_subdir="${ckpt_name%%_ckpt*}/${ckpt_setting}" ;;
  *)       ckpt_subdir="${ckpt_setting}" ;;
esac
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_subdir}"
train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
lerobot_repo_id="${OPENPI_LEROBOT_REPO_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-$(( gpu_count < 2 ? 1 : 2 ))}"

# Machine-specific knobs (caches on NFS, offline flags, lock shims, batch size,
# dataset workers) live in env_local.sh next to this script. train.sh itself stays
# generic: every knob below is an OPENPI_* env var, and sourcing env_local.sh is
# just how this machine fills them in. Values exported before calling train.sh win.
if [[ -f "${POLICY_DIR}/env_local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${POLICY_DIR}/env_local.sh" train
fi

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# JAX compilation cache always stays local for performance.
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/jax"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"

# HuggingFace datasets cache: local disk by default (upstream behavior). Set
# OPENPI_HF_HOME to place HF_HOME/HF_DATASETS_CACHE elsewhere (e.g. NFS) when the
# local disk is too small for the pyarrow mmap cache.
if [[ -n "${OPENPI_HF_HOME:-}" ]]; then
  mkdir -p "${OPENPI_HF_HOME}/datasets"
  export HF_HOME="${OPENPI_HF_HOME}"
  export HF_DATASETS_CACHE="${OPENPI_HF_HOME}/datasets"
fi

# Where openpi looks for downloaded checkpoints/assets (maybe_download reuses it).
[[ -n "${OPENPI_DATA_HOME:-}" ]] && export OPENPI_DATA_HOME

# Offline flags (HF_HUB_OFFLINE / HF_DATASETS_OFFLINE / TRANSFORMERS_OFFLINE),
# lock shims (PYTHONPATH sitecustomize + LD_PRELOAD) and the venv runner are all
# taken from the environment as-is; see env_local.sh for why this machine sets them.

echo "[Pi_05] train_config_name=${train_config_name}"
echo "[Pi_05] lerobot_repo_id=${lerobot_repo_id}"
echo "[Pi_05] fsdp_devices=${fsdp_devices}"
echo "[Pi_05] local_cache_root=${LOCAL_CACHE_ROOT}"
echo "[Pi_05] hf_datasets_cache=${HF_DATASETS_CACHE:-<default>}"
echo "[Pi_05] hf_lerobot_home=${HF_LEROBOT_HOME:-<default>}"
echo "[Pi_05] offline=HUB:${HF_HUB_OFFLINE:-0} DATASETS:${HF_DATASETS_OFFLINE:-0}"
echo "[Pi_05] ld_preload=${LD_PRELOAD:-<none>}"
echo "[Pi_05] checkpoint_dir=${ckpt_dir}"

cd "${POLICY_DIR}/openpi/"

# Runner: `uv run` (upstream default), or a synced venv python directly when
# OPENPI_VENV_PYTHON is set (uv's own NFS cache can deadlock on a broken lockd).
if [[ -n "${OPENPI_VENV_PYTHON:-}" ]]; then
  runner=("${OPENPI_VENV_PYTHON}")
else
  runner=(uv run)
fi

# Optional CLI overrides, injected only when the corresponding env var is set
# (unset = use the value baked into the TrainConfig in config.py).
extra_args=()
[[ -n "${OPENPI_WEIGHT_PATH:-}" ]] && extra_args+=(--weight-loader.params-path="${OPENPI_WEIGHT_PATH}")
[[ -n "${OPENPI_NUM_WORKERS:-}" ]] && extra_args+=(--num-workers="${OPENPI_NUM_WORKERS}")
[[ -n "${OPENPI_BATCH_SIZE:-}" ]] && extra_args+=(--batch-size="${OPENPI_BATCH_SIZE}")
[[ -n "${OPENPI_ASSETS_DIR:-}" ]] && extra_args+=(--data.assets.assets-dir="${OPENPI_ASSETS_DIR}")
[[ -n "${OPENPI_ASSET_ID:-}" ]] && extra_args+=(--data.assets.asset-id="${OPENPI_ASSET_ID}")

# openpi requires --overwrite to reuse a checkpoint dir on a fresh run, but forbids
# combining it with --resume. OPENPI_RESUME=1 continues from the latest checkpoint
# in the dir (restores params + optimizer state; data stream restarts from the top).
if [[ -n "${OPENPI_RESUME:-}" ]]; then
  extra_args+=(--resume)
else
  extra_args+=(--overwrite)
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  "${runner[@]}" scripts/train.py "${train_config_name}" \
    --exp-name="${ckpt_setting}" \
    --data.repo-id="${lerobot_repo_id}" \
    --fsdp-devices="${fsdp_devices}" \
    --checkpoint-dir-override="${ckpt_dir}" \
    --seed="${seed}" \
    "${extra_args[@]}"
