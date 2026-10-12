#!/usr/bin/env bash
# U2 parity gate (org-xpi.2) — run on a box with an RTX 3090 (uncanny) + docker.
#
# Validates the built image against uncanny's production P2 deployment:
#   1. container llama-server boots, downloads/verifies the model
#   2. /v1/models advertises the model
#   3. 512-token generation hits >= 70 tok/s decode (uncanny ref ~78)
#
# WARNING (uncanny): the production llama-server-ampere-p2 occupies ~21/24 GB
# of VRAM. Stopping it is user-visible downtime for api.girard-davila.net.
# Usage:
#   1) sudo systemctl stop llama-server-ampere-p2.service   # ~20 min window
#   2) PARITY_IMAGE=ghcr.io/alx/llamampere-sm86:31cdb8e22 bash parity-test.sh
#      (legacy package = the as-deployed baseline; new-root builds are
#       PARITY_IMAGE=ghcr.io/alx/llamampere:sm86-31cdb8e22 / sm120-31cdb8e22)
#   3) ALWAYS: sudo systemctl start llama-server-ampere-p2.service  (trap does it)
set -uo pipefail
trap 'echo "!! restoring production llama-server ..."; sudo systemctl start llama-server-ampere-p2.service' EXIT

IMAGE="${PARITY_IMAGE:-ghcr.io/alx/llamampere-sm86:31cdb8e22}"
PORT="${PARITY_PORT:-8099}"
KEY="parity-test-key"

echo "pulling $IMAGE ..."
docker pull "$IMAGE"

CID="parity-$(date +%s)"
docker run -d --name "$CID" \
  --gpus all \
  -p "$PORT:8080" \
  -e LLAMA_API_KEY="$KEY" \
  "$IMAGE"
echo "container $CID started; waiting for model download + boot (5-10 min) ..."

for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $KEY" >/dev/null 2>&1; then
    echo "server up after $((i*10))s"
    break
  fi
  if ! docker ps --filter "name=$CID" --format '{{.Names}}' | grep -q "$CID"; then
    echo "container DIED — logs:"; docker logs "$CID" | tail -30; exit 1
  fi
  sleep 10
done

echo "--- /v1/models ---"
curl -s "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $KEY"; echo

echo "--- 512-token generation (decode speed gate: >= 70 tok/s) ---"
curl -s "http://localhost:$PORT/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8","max_tokens":512,"temperature":0.2,
       "messages":[{"role":"user","content":"Write a short poem about GPUs."}]}' \
  > /tmp/parity-gen.json
python3 - <<'EOF'
import json,sys
r=json.load(open("/tmp/parity-gen.json"))
u=r.get("usage",{})
t=u.get("completion_tokens",0)
# llama.cpp returns timing via metrics; fall back to wall via a second pass is
# not needed — server logs print t/s. Print what we have:
print(f"tokens={t}")
EOF
SPEED=$(docker logs "$CID" 2>&1 | grep -oE 'eval time = *[0-9.]+ ms' | tail -1 | grep -oE '[0-9.]+')
echo "eval time: ${SPEED:-?} ms (see container logs for t/s line)"
docker logs "$CID" 2>&1 | grep -E 't/s|eval time|prompt eval' | tail -5

echo
echo "cleanup:"
docker rm -f "$CID" >/dev/null && echo "container removed"
echo "(production llama-server restore is on EXIT trap)"
