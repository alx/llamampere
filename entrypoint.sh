#!/usr/bin/env bash
# llamAmpere entrypoint for VAST instances (org-xpi U2).
# Boots the as-deployed P2 llama-server config on any RTX 3090 (sm_86).
#
# Env:
#   LLAMA_API_KEY   required — OpenAI-style API key for llama-server
#   PORT            default 8080
#   MODEL_FILE      default /models/Qwen3.8-27B-ATX-4-XS.gguf
#   HF_TOKEN        optional, if the HF repo ever requires auth
#
# Model: downloaded at boot (14.5 GiB) if missing, byte-size verified;
# a corrupt/partial download is re-fetched, never served.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-27B-ATX-4-XS.gguf}"
# jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF (as-deployed on uncanny: 15588551712 bytes)
MODEL_REPO="${MODEL_REPO:-jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF}"
MODEL_BASENAME="$(basename "$MODEL_FILE")"
EXPECTED_BYTES="${MODEL_SIZE_BYTES:-15588551712}"
PORT="${PORT:-8080}"
TMP_FILE="${MODEL_FILE}.part"

log() { echo "[entrypoint] $*" | tee -a /tmp/entrypoint.log; }

die() { echo "[entrypoint][ERROR] $*" >&2; exit 1; }

[[ -n "${LLAMA_API_KEY:-}" ]] || die "LLAMA_API_KEY env var is required"

# --- model download + verify --------------------------------------------
need_download=1
if [[ -f "$MODEL_FILE" ]]; then
  actual="$(stat -c %s "$MODEL_FILE")"
  if [[ "$actual" == "$EXPECTED_BYTES" ]]; then
    need_download=0
  else
    log "model file present but size $actual != expected $EXPECTED_BYTES — re-downloading"
    rm -f "$MODEL_FILE"
  fi
fi

if [[ "$need_download" == "1" ]]; then
  mkdir -p "$MODEL_DIR"
  log "downloading $MODEL_REPO/$MODEL_BASENAME (~14.5 GiB) ..."
  url="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_BASENAME}?download=true"
  # --retry + partial resume; HF 401/404 -> non-zero exit -> die (never serve partial)
  curl -fL --retry 3 --retry-delay 5 -C - -H "Authorization: Bearer ${HF_TOKEN:-none}" \
    "$url" -o "$TMP_FILE" \
    || { rm -f "$TMP_FILE"; die "model download failed (check HF_TOKEN / repo name)"; }
  actual="$(stat -c %s "$TMP_FILE")"
  [[ "$actual" == "$EXPECTED_BYTES" ]] || { rm -f "$TMP_FILE"; die "model size mismatch: got $actual, want $EXPECTED_BYTES"; }
  mv "$TMP_FILE" "$MODEL_FILE"
  log "model verified ($actual bytes)"
fi

# --- RAM-aware flags -----------------------------------------------------
# uncanny box: 16 GiB RAM -> 12 checkpoints @ 20480 + --cache-ram 4096 (P2).
# Small-RAM VAST instances (< 32 GiB): drop the RAM-heavy replay features.
TOTAL_RAM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
CTX=237000
EXTRA_FLAGS=()
if (( TOTAL_RAM_GB < 32 )); then
  log "RAM $TOTAL_RAM_GB GiB < 32 — dropping ctx-checkpoints/cache-ram, ctx 237k->180k"
  CTX=180000
else
  log "RAM $TOTAL_RAM_GB GiB — full P2 flags (checkpoints + cache-ram)"
  EXTRA_FLAGS+=(--cache-prompt --cache-ram 4096 --ctx-checkpoints 12 --checkpoint-min-step 20480)
fi

log "booting llama-server: ctx=$CTX ram=${TOTAL_RAM_GB}GiB port=$PORT (commit ${LLAMAAMPRE_COMMIT:-?})"
exec /opt/llama/bin/llama-server \
  -m "$MODEL_FILE" \
  -a qwen3.8 \
  -ngl 99 --host 0.0.0.0 --port "$PORT" -np 1 \
  -c "$CTX" --flash-attn on \
  --batch-size 4096 --ubatch-size 1024 \
  --cache-type-k q8_0 --cache-type-v turbo3 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --spec-draft-vocab-map /opt/llama/vocab/atx_65536.txt \
  --fit off \
  "${EXTRA_FLAGS[@]}" \
  --metrics --perf \
  --threads 8 --threads-batch 8 \
  --jinja --chat-template-file /opt/llama/templates/Qwen3.8-Sharp.jinja \
  --chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --no-mmproj-offload \
  --api-key "$LLAMA_API_KEY"
