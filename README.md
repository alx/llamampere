# llamampere-docker (VAST.ai inference images)

Docker packaging for the **llamAmpere** llama.cpp fork — prebuilt per GPU
architecture and serving **Qwen3.8-27B-ATX-IQ4_XS** on VAST.ai behind the
`api.girard-davila.net` LiteLLM proxy.

> **Upstream / root project:** [`github.com/JakeATX/llamAmpere`](https://github.com/JakeATX/llamAmpere)
> — this repo contains **no source of the fork itself**, only the Docker build,
> entrypoint and assets. Builds default to the **latest upstream `main`** commit
> (resolved at build time; the tag carries the sha); pin with
> `LLAMAAMPRE_COMMIT=<sha>`.

All archs live in ONE GHCR package, `ghcr.io/alx/llamampere-docker`, with the
arch in the tag.

The image is designed to boot on any 3090-class VAST instance (16 GiB+ RAM). It downloads the
14.5 GiB model from Hugging Face at first boot (size-verified, re-fetched if corrupt), then serves
OpenAI-compatible inference on port 8080.

## Targets (two tiers of the self-hosted offer, org-or6b.12)

One package, common root, arch as the tag dimension:

| Target | Dockerfile | Tag | GPU | Tier |
|---|---|---|---|---|
| `sm86` (default) | `Dockerfile` | `ghcr.io/alx/llamampere-docker:sm86-<commit>` | RTX 3090 24 GB (CUDA 12.4) | standard |
| `sm120` | `Dockerfile.sm120` | `ghcr.io/alx/llamampere-docker:sm120-<commit>` | RTX 5090 32 GB (CUDA 12.8.1) | premium |

The legacy per-arch package `ghcr.io/alx/llamampere-sm86` (tag `31cdb8e22`, upstream
`31cdb8e` = release/v0.3) stays as the as-deployed baseline for the running standard-tier
template; the next standard rebuild moves to the new root.

Same upstream commit, same entrypoint/assets/model in both tiers; `sm120` differs only in the
CUDA base (Blackwell kernels need >= 12.8), the arch flag (`120a`), and a version-agnostic
cublas stage. sm_90 (Hopper) is deliberately NOT built: DC H100/H200 rents at $2.59–5.79/hr
(10–17× a 3090) and only pays off for 80–140 GB model classes.

## Server profile (`SERVER_PROFILE`, org-or6b.4)

`entrypoint.sh` branches on the baked-in `SERVER_PROFILE` (build arg, default
`qwen38`):

| Profile | What it serves | Flags |
|---|---|---|
| `qwen38` (default) | Qwen3.8-27B ATX (production P2 config) | MTP spec decode, turbo3 V-cache, 237k/180k ctx, Qwen jinja template, byte-verified 15588551712-byte download |
| `llama8` | 8B test mule (multi-KV tenancy experiments) | plain: no MTP/vocab-map/turbo3, 120k/96k ctx, llama.cpp built-in chat template; **no default `MODEL_REPO`** — the 8B GGUF pick (and its license) is an org-or6b.4 decision, so the entrypoint dies asking for `MODEL_REPO` instead of guessing |

`PROFILE=llama8 ./build-and-push.sh` bakes the profile in and suffixes the tag
(`sm86-<commit>-llama8`). Non-default env (`MODEL_REPO`, `MODEL_FILE`,
`MODEL_SIZE_BYTES`) still override per-instance at launch.

## What's in the image

- `llama-server` + the full shared-lib closure (`libllama*`, `libggml*`, `libmtmd`) in
  `/opt/llama/bin` — built with `BUILD_SHARED_LIBS=ON`, sm_86, CUDA 12.4.
- `libcublas` / `libcublasLt` 12.4 runtime copied from the CUDA devel builder stage.
- NCCL from the `nvidia-nccl-cu12` pip wheel (lib-only, pip discarded).
- P2 runtime assets: `templates/Qwen3.8-Sharp.jinja`, `vocab/atx_65536.txt`.
- Runtime base `nvidia/cuda:12.4.1-base-ubuntu22.04` (~1.9 GB total, no CUDA dev toolchain).

## Build

```sh
./build-and-push.sh              # standard tier -> ghcr.io/alx/llamampere-docker:sm86-<commit>
TARGET=sm120 ./build-and-push.sh # premium tier  -> ghcr.io/alx/llamampere-docker:sm120-<commit>
# or manually:
docker build -f Dockerfile --build-arg LLAMAAMPRE_COMMIT=<sha> -t ghcr.io/alx/llamampere-docker:sm86-<short> .
docker push ghcr.io/alx/llamampere-docker:sm86-<short>
```

After a successful push, record the printed digest in `vast/config.env` on the deploy box
(`IMAGE_DIGEST` for sm86, `IMAGE_DIGEST_SM120` for sm120 — digest-pinned, tag-agnostic).

The GHCR package page renders this README because the image carries
`LABEL org.opencontainers.image.source=https://github.com/alx/llamampere-docker`.

## Deploy (VAST.ai)

The image is launched via a private VAST template. On the deploy box (`lamai270`):

```sh
cd ~/code/api.girard-davila.net/vast
.venv/bin/python -m scripts.launch --yes   # picks top 3090 offer, launches, prints endpoint
.venv/bin/python -m scripts.status         # instances, rate, spend
.venv/bin/python -m scripts.teardown --id <instance-id>
```

`config.env` (deploy box) pins:

- `IMAGE_DIGEST=ghcr.io/alx/llamampere-sm86@sha256:<manifest-digest>` (as-deployed legacy pin)
- `IMAGE_DIGEST_SM120=ghcr.io/alx/llamampere-docker@sha256:<manifest-digest>` (premium tier)
- `TEMPLATE_ID` / `TEMPLATE_HASH_ID` (standard, created by `scripts.create_template --tier standard`)
- `TEMPLATE_ID_SM120` / `TEMPLATE_HASH_ID_SM120` (premium, `--tier premium`)

`scripts.launch --tier standard|premium` (or `VAST_TIER`) selects the offer class and the
template pin; `scripts.create_template --tier <t>` creates/re-points that tier's template.
- `VAST_API_KEY`

The per-launch `LLAMA_API_KEY` is injected at `create_instance` time; it is **not** baked into the
template.

## Runtime flags (P2)

Set in `entrypoint.sh`: 180k ctx (< 32 GiB RAM) or 237k ctx + 12 checkpoints (32 GiB+),
MTP-2 speculative decode, turbo3 K/V cache, flash attention, q8_0 K-cache, threads 8.
