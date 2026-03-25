#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

run_pip() {
  "${PYTHON_BIN}" -m pip "$@"
}

prompt_choice() {
  local prompt="$1"
  local value
  read -r -p "${prompt}" value
  echo "${value}" | tr '[:upper:]' '[:lower:]'
}

echo "Step 1/4: Installing lite (base) requirements..."
run_pip install -e .
echo "Lite install complete."

echo
echo "Step 2/4: Optional ML backend"
echo "Choose one:"
echo "  [none] Skip ML dependencies"
echo "  [cpu]  Install CPU-only torch + open-clip-torch"
echo "  [gpu]  Install GPU torch command you provide + open-clip-torch"
ml_choice="$(prompt_choice "ML choice [none/cpu/gpu]: ")"

case "${ml_choice}" in
  none|"")
    echo "Skipping ML dependency installation."
    ;;
  cpu)
    run_pip install torch --index-url https://download.pytorch.org/whl/cpu
    run_pip install open-clip-torch
    echo "CPU ML dependencies installed."
    ;;
  gpu)
    echo "Enter the PyTorch index URL for your CUDA version."
    echo "Example (CUDA 12.1): https://download.pytorch.org/whl/cu121"
    echo "Find yours at: https://pytorch.org/get-started/locally/"
    read -r -p "PyTorch index URL: " torch_index_url

    # Validate: must be a non-empty https:// URL pointing to download.pytorch.org
    if [[ -z "${torch_index_url}" ]]; then
      echo "No URL provided. Skipping GPU torch install."
    elif [[ "${torch_index_url}" != https://download.pytorch.org/* ]]; then
      echo "Error: URL must start with 'https://download.pytorch.org/'."
      echo "Skipping GPU torch install for safety."
    else
      run_pip install torch --index-url "${torch_index_url}"
      run_pip install open-clip-torch
      echo "GPU ML dependencies installed."
    fi
    ;;
  *)
    echo "Unknown choice '${ml_choice}'. Skipping ML dependency installation."
    ;;
esac

echo
echo "Step 3/4: Optional video/face dependencies"
video_choice="$(prompt_choice "Install video extra? [y/N]: ")"
if [[ "${video_choice}" == "y" || "${video_choice}" == "yes" ]]; then
  run_pip install -e .[video]
  echo "Video/face dependencies installed."
else
  echo "Skipping video/face dependencies."
fi

echo
echo "Step 4/4: Optional finalize/export dependencies"
finalize_choice="$(prompt_choice "Install finalize extra? [y/N]: ")"
if [[ "${finalize_choice}" == "y" || "${finalize_choice}" == "yes" ]]; then
  run_pip install -e .[finalize]
  echo "Finalize/export dependencies installed."
else
  echo "Skipping finalize/export dependencies."
fi

echo
echo "Tip: run 'media-sorter doctor --expect-finalize --expect-video' to verify the environment."
echo
echo "Setup complete."
