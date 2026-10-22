# env_local.sh — machine-specific knobs for this box (L20 node, NFS-backed storage).
#
# Usage: source env_local.sh [train|eval]      (no arg = both)
#   train.sh sources it as `train`, eval.sh as `eval`.
#
# Every value is an ordinary environment variable: exporting a different value
# before calling train.sh / eval.sh wins. Kept out of upstream; committed to the
# personal fork so the machine setup is backed up and documented.

_env_local_mode="${1:-all}"
ENV_LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------- both train and eval ---
# pi05_base weights were already downloaded here; OPENPI_DATA_HOME makes
# maybe_download reuse the local copy instead of re-fetching 12G from GCS.
# (train: CheckpointWeightLoader; eval: picked up by setup_eval_policy_server.sh)
export OPENPI_DATA_HOME="/mnt/nfs01/whl/home_cache/openpi"

# ---------------------------------------------------------------- train only ---
if [[ "${_env_local_mode}" == "train" || "${_env_local_mode}" == "all" ]]; then

  # Dataset images decode on CPU; 8 workers starve the GPUs on this 64-core box.
  export OPENPI_NUM_WORKERS=32
  # Old *_seed_* configs default to batch_size=256 / fsdp_devices=2 (4-GPU box);
  # this box trains on 1-2x L20.
  export OPENPI_BATCH_SIZE=32
  export OPENPI_FSDP_DEVICES=1
  # Old configs point assets at str(_ROBODOJO_ASSETS_DIR); this box loads norm
  # stats / assets from the public pi05_base bundle instead.
  export OPENPI_ASSETS_DIR="gs://openpi-assets/checkpoints/pi05_base/assets"
  export OPENPI_ASSET_ID="arx"

  # `uv run` deadlocks on this cluster: uv's cache sits on NFS and uv takes native
  # advisory locks (not Python filelock) that hang on the broken NFS lockd. Use the
  # already-synced venv python directly instead.
  export OPENPI_VENV_PYTHON="${ENV_LOCAL_DIR}/openpi/.venv/bin/python"

  # HF datasets pyarrow mmap cache goes to NFS (local disk too small); JAX
  # compilation cache stays local (train.sh handles it, faster on local disk).
  export OPENPI_HF_HOME="/mnt/nfs01/whl/home_cache/huggingface"
  # Converted LeRobot datasets live here.
  export OPENPI_HF_LEROBOT_HOME="/mnt/nfs01/whl/robodojo_conversion/.cache/huggingface/lerobot"

  # Offline mode: datasets come from NFS, never touch huggingface.co.
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  # NFS lockd (NLM) on this cluster is broken: flock()/fcntl() hang uninterruptibly
  # in kernel D state (rpc_wait_bit_killable). Two workarounds, both safe because a
  # training run is the sole writer:
  # 1) sitecustomize.py transparently redirects Python filelock files under /mnt/nfs*
  #    to local disk (data stays on NFS; only the tiny lock file moves).
  # 2) The redirect does NOT cover tensorstore (orbax checkpoint writer), which takes
  #    fcntl(F_OFD_SETLK) locks in C++ and hangs forever; nolock_shim.so neutralizes
  #    advisory fcntl/flock locks (passes all other cmds through).
  export PYTHONPATH="${HOME}/openpi_data_local/pybootstrap${PYTHONPATH:+:${PYTHONPATH}}"
  export LD_PRELOAD="${HOME}/openpi_data_local/pybootstrap/nolock_shim.so${LD_PRELOAD:+:${LD_PRELOAD}}"

fi

# ----------------------------------------------------------------- eval only ---
if [[ "${_env_local_mode}" == "eval" || "${_env_local_mode}" == "all" ]]; then

  : # eval-specific knobs go here if ever needed

fi

# ------------------------------------------- per-run switches (both modes) -----
# Uncomment per run, or export before calling train.sh / eval.sh:
# export OPENPI_WEIGHT_PATH=/mnt/nfs01/whl/Pi_05_checkpoint/RoboDojo-sim-arx_x5-joint-0/59999/params
# export EVAL_BATCH_OVERRIDE=false    # collection mode (deploy.yml default is true)
# export POLICY_CHECKPOINT_NUM=5999   # override checkpoint_num from deploy.yml
# ROBODOJO_SCENE_SEED is set by eval.sh itself (11th positional arg).

unset _env_local_mode
