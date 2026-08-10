#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

nvcc --version | grep -q "release 13.3" || fail "nvcc is not 13.3 (13.1 breaks on 26.04 glibc)"
nvcc --list-gpu-arch | grep -q compute_120a || fail "nvcc lacks compute_120a"
nvidia-smi --query-gpu=driver_version --format=csv,noheader

source "$VLLM_VENV/bin/activate"
python - <<'EOF'
import torch, sys
cap = torch.cuda.get_device_capability()
assert cap == (12, 0), f"expected sm_120, got {cap}"
print(f"torch {torch.__version__} cuda {torch.version.cuda} {cap}")
EOF

fi_versions=$(uv pip list | awk '/^flashinfer/ {print $2}' | sort -u | wc -l)
[ "$fi_versions" -eq 1 ] || fail "flashinfer version skew: $(uv pip list | grep flashinfer)"

python -c "import vllm; print(vllm.__file__, vllm.__version__)"
echo "OK"

