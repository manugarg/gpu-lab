export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_VENV="${VLLM_VENV:-$HOME/venvs/vllm}"

export MODEL="${MODEL:-Qwen/Qwen3-14B-FP8}"
export SERVE_ARGS="${SERVE_ARGS:---attention-backend flashinfer --gpu-memory-utilization 0.90 --max-model-len 16384}"

# llama.cpp (see serve/README.md). Only used by serve/llama.sh.
export LLAMA_BIN="${LLAMA_BIN:-$HOME/tools/llama.cpp/build/bin/llama-server}"
export LLAMA_GGUF_REPO="${LLAMA_GGUF_REPO:-models--unsloth--Qwen3.8-27B-GGUF}"
export LLAMA_GGUF_FILE="${LLAMA_GGUF_FILE:-Qwen3.8-27B-UD-Q5_K_XL.gguf}"
