export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"

export MODEL="${MODEL:-Qwen/Qwen3-14B-FP8}"
export SERVE_ARGS="${SERVE_ARGS:---attention-backend flashinfer --gpu-memory-utilization 0.90 --max-model-len 16384}"
