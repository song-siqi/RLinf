#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
SRC_FILE="${SCRIPT_DIR}/train_vla_sft.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_PATH}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-x2robot_s2m_sft_openpi_8gpu}"

export EMBODIED_PATH="${SCRIPT_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/public/songsiqi/data/lerobot}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-online}"

LOG_DIR="${REPO_PATH}/logs/x2robot_s2m_sft_$(date +"%Y%m%d-%H%M%S")"
mkdir -p "${LOG_DIR}"

echo "Using Python: ${PYTHON_BIN}"
echo "Using config: ${CONFIG_NAME}"
echo "HF_LEROBOT_HOME: ${HF_LEROBOT_HOME}"
echo "Log dir: ${LOG_DIR}"

"${PYTHON_BIN}" "${SRC_FILE}" \
  --config-path "${SCRIPT_DIR}/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  "$@" 2>&1 | tee -a "${LOG_DIR}/run_sft.log"
