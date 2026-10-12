# llamAmpere sm_86 (RTX 3090)

llama.cpp fork ("llamAmpere", commit `31cdb8e2`) prebuilt for **NVIDIA RTX 3090 (sm_86)**, serving
**Qwen3.8-27B-ATX-IQ4_XS** on VAST.ai behind the `api.girard-davila.net` LiteLLM proxy.

The image is designed to boot on any 3090-class VAST instance (16 GiB+ RAM). It downloads the
14.5 GiB model from Hugging Face at first boot (size-verified, re-fetched if corrupt), then serves
OpenAI-compatible inference on port 8080.

## Targets (two tiers of the self-hosted offer, org-or6b.12)

| Target | Dockerfile | GHCR package | GPU | Tier |
|---|---|---|---|---|
| `sm86` (default) | `Dockerfile` | `ghcr.io/alx/llamampere-sm86` | RTX 3090 24 GB (CUDA 12.4) | standard |
| `sm120` | `Dockerfile.sm120` | `ghcr.io/alx/llamampere-sm120` | RTX 5090 32 GB (CUDA 12.8.1) | premium |

Same pinned llamAmpere commit, same entrypoint/assets/model in both; `sm120` differs only in the
CUDA base (Blackwell kernels need >= 12.8), the arch flag (`120a`), and a version-agnostic
cublas stage. sm_90 (Hopper) is deliberately NOT built: DC H100/H200 rents at $2.59–5.79/hr
(10–17× a 3090) and only pays off for 80–140 GB model classes.

## What's in the image

- `llama-server` + the full shared-lib closure (`libllama*`, `libggml*`, `libmtmd`) in
  `/opt/llama/bin` — built with `BUILD_SHARED_LIBS=ON`, sm_86, CUDA 12.4.
- `libcublas` / `libcublasLt` 12.4 runtime copied from the CUDA devel builder stage.
- NCCL from the `nvidia-nccl-cu12` pip wheel (lib-only, pip discarded).
- P2 runtime assets: `templates/Qwen3.8-Sharp.jinja`, `vocab/atx_65536.txt`.
- Runtime base `nvidia/cuda:12.4.1-base-ubuntu22.04` (~1.9 GB total, no CUDA dev toolchain).

## Build

```sh
./build-and-push.sh              # standard tier (sm86)
TARGET=sm120 ./build-and-push.sh # premium tier (sm120 / RTX 5090)
# or manually:
docker build -f Dockerfile --build-arg LLAMAAMPRE_COMMIT=<sha> -t ghcr.io/alx/llamampere-sm86:<tag> .
docker push ghcr.io/alx/llamampere-sm86:<tag>
```

After a successful push, record the printed digest in `vast/config.env` on the deploy box
(`IMAGE_DIGEST` for sm86, `IMAGE_DIGEST_SM120` for sm120).

The GHCR package page renders this README because the image carries
`LABEL org.opencontainers.image.source=https://github.com/alx/llamampere-sm86`.

## Deploy (VAST.ai)

The image is launched via a private VAST template. On the deploy box (`lamai270`):

```sh
cd ~/code/api.girard-davila.net/vast
.venv/bin/python -m scripts.launch --yes   # picks top 3090 offer, launches, prints endpoint
.venv/bin/python -m scripts.status         # instances, rate, spend
.venv/bin/python -m scripts.teardown --id <instance-id>
```

`config.env` (deploy box) pins:

- `IMAGE_DIGEST=ghcr.io/alx/llamampere-sm86@sha256:<manifest-digest>`
- `TEMPLATE_ID` / `TEMPLATE_HASH_ID` (created by `scripts.create_template`)
- `VAST_API_KEY`

The per-launch `LLAMA_API_KEY` is injected at `create_instance` time; it is **not** baked into the
template.

## Runtime flags (P2)

Set in `entrypoint.sh`: 180k ctx (< 32 GiB RAM) or 237k ctx + 12 checkpoints (32 GiB+),
MTP-2 speculative decode, turbo3 K/V cache, flash attention, q8_0 K-cache, threads 8.
