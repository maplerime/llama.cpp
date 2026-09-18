#!/usr/bin/env bash
# Start llama-server in the background with WSTREAM remote memory + hot-expert cache.
# Weights live in the remote memserver pool (cathy-mem0); the local GPU does compute.
# Log goes to server.log next to this script. Edit the vars below to change setup.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

# --- model (largest available: Qwen3-235B-A22B, 141 GB of weights held remotely) ---
MODEL="models/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf"
# Smaller/faster option (30B, ~5 t/s with cache):
# MODEL="models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"

# --- remote memory server ---
export GGML_CUDA_WSTREAM=1
export GGML_CUDA_WSTREAM_REMOTE=1
export GGML_CUDA_WSTREAM_HOST=10.244.64.5
export GGML_CUDA_WSTREAM_PORT=9797

# --- VRAM budgets ---
export GGML_CUDA_WSTREAM_VRAM=4000000000       # pinned non-expert weights
export GGML_CUDA_WSTREAM_MIN=1073741824        # min buffer size treated as streamable
export GGML_CUDA_WSTREAM_ECACHE_M=1            # 1 = enable hot-expert resident cache
export GGML_CUDA_WSTREAM_ECACHE=15000000000    # cache budget in bytes (15 GiB)

# --- server ---
HOST=0.0.0.0
PORT=8080
CTX=4096
LOG="$PWD/server.log"

setsid ./build/bin/llama-server \
  -m "$MODEL" \
  -ngl 999 -c "$CTX" \
  --host "$HOST" --port "$PORT" \
  </dev/null >"$LOG" 2>&1 &

echo "llama-server starting in background (pid $!)"
echo "model: $MODEL"
echo "log:   $LOG   (tail -f to watch)"
echo "test:  curl -s http://127.0.0.1:$PORT/health"
