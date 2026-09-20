#!/usr/bin/env bash
# Start llama-server in the background with WSTREAM remote memory + hot-expert cache.
# Weights live in the remote memserver pool (cathy-mem0); the local GPU does compute.
# Log goes to server.log next to this script. Edit the vars below to change setup.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

# --- model (largest: DeepSeek-V3-0324 671B, ~247 GB of weights held remotely) ---
MODEL="models/UD-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf"
# Other options:
# MODEL="models/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf"   # 235B
# MODEL="models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"                      # 30B, fast

# --- remote memory server ---
export GGML_CUDA_WSTREAM=1
export GGML_CUDA_WSTREAM_REMOTE=1
export GGML_CUDA_WSTREAM_HOST=10.244.64.5
export GGML_CUDA_WSTREAM_PORT=9797

# --- VRAM budgets ---
export GGML_CUDA_WSTREAM_VRAM=1000000000       # pinned non-expert weights (minimal)
export GGML_CUDA_WSTREAM_MIN=1073741824        # min buffer size treated as streamable
export GGML_CUDA_WSTREAM_COMPACT_M=0           # 0 = pure streaming (fit first, add cache after it runs)
# (compact is decode-only + remap; fits 671B where full-size resident cannot.
#  For a model that fits VRAM full-size, use GGML_CUDA_WSTREAM_ECACHE_M=1 + _ECACHE=<bytes> instead.)

# --- server ---
HOST=0.0.0.0
PORT=8080
CTX=1024          # small KV/attention buffers on the 22 GB L4
BATCH=32
UBATCH=32         # tiny prefill ubatch: minimal compute buffers
PARALLEL=1        # single slot
LOG="$PWD/server.log"

setsid ./build/bin/llama-server \
  -m "$MODEL" \
  -ngl 999 -c "$CTX" -b "$BATCH" -ub "$UBATCH" --parallel "$PARALLEL" \
  --host "$HOST" --port "$PORT" \
  </dev/null >"$LOG" 2>&1 &

echo "llama-server starting in background (pid $!)"
echo "model: $MODEL"
echo "log:   $LOG   (tail -f to watch; wait for 'server is listening')"
echo "health: curl -s http://127.0.0.1:$PORT/health"
echo ""
echo "IMPORTANT: use the CHAT endpoint so the model's chat template is applied."
echo "Raw /completion feeds the model unformatted text and it just echoes/rambles"
echo "(this is not a bug). Correct call:"
echo ""
curl -s http://127.0.0.1:$PORT/v1/chat/completions -d '{"messages":[{"role":"user","content":"你的问题"}], "max_tokens":128, "temperature":0.6 }'
echo "(671B remote is slow, ~0.06 t/s pure-streaming; be patient per request.)"
