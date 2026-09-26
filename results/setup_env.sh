#!/usr/bin/env bash
# RQ1b -- environment for the authors' RiskProp code (MMAction2 1.2.0 / mmcv 2.2.0 /
# mmengine 0.10.7, as in their README). Works on any vast.ai image: it builds its own
# Python 3.10 venv with uv, because mmcv 2.2.0 and torch 2.1 have no Python 3.12 wheels.
#
# Usage:  bash /workspace/CPV301/riskprop_full/setup_env.sh
# Then in every new terminal:  source /workspace/rq1b_env/bin/activate
set -euo pipefail
WS=/workspace
CPV=$WS/CPV301
RP=$WS/RiskProp
RISKPROP_COMMIT=579376fa1d879f28a9d68c2565d1e449de24b666

python3 -m pip install -q uv || pip install -q uv
uv venv "$WS/rq1b_env" -p 3.10
# shellcheck disable=SC1091
source "$WS/rq1b_env/bin/activate"

echo "== PyTorch 2.1.2 + CUDA 12.1"
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "== data / analysis libraries (numpy 1.x for torch 2.1)"
uv pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84" "pandas==2.2.3" \
    "scikit-learn==1.5.2" "scipy==1.13.1" "matplotlib==3.9.2" decord einops openpyxl xlrd \
    pillow huggingface_hub fvcore iopath tqdm

echo "== OpenMMLab (versions from the authors' README)"
uv pip install "mmengine==0.10.7"
if ! uv pip install "mmcv==2.2.0" --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html; then
    echo "!! mmcv 2.2.0 wheel not found for torch2.1/cu121 -- falling back to 2.1.0 (allowed by the"
    echo "!! authors' version check 2.0.0rc4 <= mmcv <= 2.2.0). Write this down as a deviation."
    uv pip install "mmcv==2.1.0" --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
fi

echo "== authors' code at commit ${RISKPROP_COMMIT:0:7}"
if [ ! -d "$RP/.git" ]; then
    git clone https://github.com/xingyueye5/RiskProp "$RP"
fi
git -C "$RP" checkout -q "$RISKPROP_COMMIT"
uv pip install --no-deps -e "$RP"
python "$CPV/riskprop_full/patch_authors_code.py" "$RP"
cp "$CPV/riskprop_full/configs/riskprop_full_nexar.py" "$RP/configs/"

echo "== check"
export PYTHONPATH="$CPV/riskprop_full:${PYTHONPATH:-}"
python - <<'EOF'
import torch, mmcv, mmengine, mmaction, taa, nexar_rq1b
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("mmcv", mmcv.__version__, "| mmengine", mmengine.__version__, "| mmaction", mmaction.__version__)
from mmaction.registry import MODELS, DATASETS
assert "AnticipationHead" in MODELS.module_dict and "NexarRQ1BDataset" in DATASETS.module_dict
assert torch.cuda.is_available(), "no GPU visible"
print("OK")
EOF
uv pip freeze > "$CPV/riskprop_full/environment_rq1b.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$CPV/riskprop_full/environment_rq1b.txt"
echo "Environment ready. Package list: $CPV/riskprop_full/environment_rq1b.txt"
