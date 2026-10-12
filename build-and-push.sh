#!/usr/bin/env bash
# Build + push a llamAmpere image to GHCR and print the digest (org-xpi U2).
#
# Targets (org-or6b.12: two tiers of the qwen3.8-27b self-hosted offer):
#   TARGET=sm86   (default)  Dockerfile,        ghcr.io/alx/llamampere-sm86    — standard tier (RTX 3090)
#   TARGET=sm120              Dockerfile.sm120, ghcr.io/alx/llamampere-sm120   — premium tier (RTX 5090, CUDA 12.8)
#
# Run FROM ANY x86_64 DOCKER BOX (fami is the reference build box): the CUDA
# toolkit comes from the base image, so no local GPU/nvcc is needed — only the
# parity test needs a real GPU of the target arch. Requires: docker buildx,
# gh auth (GHCR push).
#
# Usage (from the repo):
#   BUILD_JOBS=2 bash build-and-push.sh                 # BUILD_JOBS caps -j
#   TARGET=sm120 bash build-and-push.sh                # premium tier
#   GHCR_IMAGE=... bash build-and-push.sh              # override target
#
# Rebuilds are idempotent: bump LLAMAAMPRE_COMMIT in this script (and the
# Dockerfile ARG default) only if the llamAmpere pin changes. ~30-60 min for a
# cold compile; buildx caches per-layer, so a Dockerfile tweak re-runs only the
# changed layers.
#
# After success, record the printed digest in vast/config.env:
#   sm86  -> IMAGE_DIGEST=ghcr.io/alx/llamampere-sm86@sha256:...
#   sm120 -> IMAGE_DIGEST_SM120=ghcr.io/alx/llamampere-sm120@sha256:...
set -euo pipefail

IMAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # build context
CTX="$IMAGE_DIR"
TARGET="${TARGET:-sm86}"
if [[ -n "${GHCR_IMAGE:-}" ]]; then
    : # explicit override wins
elif [[ "$TARGET" == "sm120" ]]; then
    GHCR_IMAGE="ghcr.io/alx/llamampere-sm120:31cdb8e22"
    DOCKERFILE="Dockerfile.sm120"
else
    [[ "$TARGET" == "sm86" ]] || { echo "TARGET must be sm86|sm120, got: $TARGET" >&2; exit 1; }
    GHCR_IMAGE="ghcr.io/alx/llamampere-sm86:31cdb8e22"
    DOCKERFILE="Dockerfile"
fi
COMMIT="31cdb8e2257aeb855462d1697e68b14738818f49"
SHORT="${COMMIT:0:7}"
# CUDA kernel builds are RAM-hungry; cap parallel jobs (raise on beefier boxes).
BUILD_JOBS="${BUILD_JOBS:-4}"

# --- preflight ------------------------------------------------------------
command -v gh >/dev/null || { echo "need gh (GHCR auth)" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated" >&2; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "need docker buildx" >&2; exit 1; }
chmod +x "$IMAGE_DIR/entrypoint.sh"

# GHCR push auth via gh token
gh auth token | docker login ghcr.io -u "$(gh api user -q .login)" --password-stdin

# --- build (push directly: never materializes the image in local disk) -----
echo "Building + pushing $GHCR_IMAGE ..."
docker buildx build \
  --platform linux/amd64 \
  --file "$CTX/$DOCKERFILE" \
  --build-arg LLAMAAMPRE_COMMIT="$COMMIT" \
  --build-arg BUILD_JOBS="$BUILD_JOBS" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --push \
  --progress plain \
  -t "$GHCR_IMAGE" \
  "$CTX"

# --- record digest (registry API, gh token) --------------------------------
# Give the registry a few seconds to publish the manifest after push.
sleep 5
# repo path = <image> with host + :tag removed  ->  alx/llamampere-sm86
REPO_PATH="${GHCR_IMAGE#ghcr.io/}"
REPO_PATH="${REPO_PATH%%:*}"
GHTOKEN="$(gh auth token)"
# anonymous GHCR token for pull-scoped read on this repo
RTOKEN="$(curl -s "https://ghcr.io/token?scope=repository:${REPO_PATH}:pull" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))')"
DIGEST="$(curl -sI \
  -H "Authorization: Bearer ${RTOKEN:-$GHTOKEN}" \
  -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json' \
  "https://ghcr.io/v2/${REPO_PATH}/manifests/${SHORT}" \
  | grep -i '^docker-content-digest' | awk '{print $2}' | tr -d '\r')"

echo
echo "=== DIGEST (record in vast/config.env) ==="
if [[ -n "${DIGEST:-}" ]]; then
  echo "IMAGE_DIGEST=${GHCR_IMAGE%%:*}@${DIGEST}"
else
  echo "[warn] could not read digest from registry; resolve manually:"
  echo "  docker buildx imagetools inspect $GHCR_IMAGE"
fi
