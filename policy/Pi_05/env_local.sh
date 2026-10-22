# env_local.sh — machine-specific knobs for this box (L20 node, NFS-backed storage).
#
# Usage: source env_local.sh [train|eval]      (no arg = both)
#   eval.sh sources it as `eval`.
#
# Every value is an ordinary environment variable: exporting a different value
# before calling eval.sh wins. Kept out of upstream; committed to the personal
# fork so the machine setup is backed up and documented.

_env_local_mode="${1:-all}"
ENV_LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------- both train and eval ---
# pi05_base weights were already downloaded here; OPENPI_DATA_HOME makes
# maybe_download reuse the local copy instead of re-fetching 12G from GCS.
# (eval: picked up by setup_eval_policy_server.sh)
export OPENPI_DATA_HOME="/mnt/nfs01/whl/home_cache/openpi"

# ----------------------------------------------------------------- eval only ---
if [[ "${_env_local_mode}" == "eval" || "${_env_local_mode}" == "all" ]]; then

  : # eval-specific knobs go here if ever needed

fi

# ------------------------------------------- per-run switches (both modes) -----
# Uncomment per run, or export before calling eval.sh:
# export EVAL_BATCH_OVERRIDE=false    # collection mode (deploy.yml default is true)
# export POLICY_CHECKPOINT_NUM=5999   # override checkpoint_num from deploy.yml
# ROBODOJO_SCENE_SEED is set by eval.sh itself (11th positional arg).

unset _env_local_mode
