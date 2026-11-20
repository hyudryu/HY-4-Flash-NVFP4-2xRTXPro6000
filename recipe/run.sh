#!/usr/bin/env bash
# Serve Avifenesh/Hy4-flash-NVFP4 on 2x RTX PRO 6000 Blackwell (SM120) with vLLM,
# using the vendor FP8 DFly drafter from tencent/Hy4-flash-FP8.
# The stock vllm/vllm-openai:hy4-flash image supports neither SM120 for this model
# nor this NVFP4 checkpoint; the seven files in ../patch are bind-mounted read-only
# over the image (see README "The patch").
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
P="$REPO/patch/vllm/models"
D=/usr/local/lib/python3.12/dist-packages/vllm/models
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:6b645bfbf4b89602d38f6e93c1c418516ec0bfbf54b64cd661b8075bfec01da1}  # tag hy4-flash
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
PORT=${PORT:-8000}

MOUNTS=()
for f in hy_v4_flash/nvidia/sm120.py hy_v4_flash/nvidia/attention.py hy_v4/nvidia/attention.py \
         hy_v4_flash/nvidia/model.py hy_v4_flash/nvidia/engram.py hy_v4_flash/nvidia/nvfp4_fold.py \
         hy_v4/dfly_config.py; do
  MOUNTS+=(-v "$P/$f:$D/$f:ro")
done

docker rm -f hy4-flash-nvfp4 >/dev/null 2>&1 || true
docker run -d --name hy4-flash-nvfp4 --gpus all --ipc=host --shm-size 16g -p "$PORT:8000" \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_RUST_FRONTEND=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "${MOUNTS[@]}" \
  --entrypoint vllm "$IMAGE" \
  serve Avifenesh/Hy4-flash-NVFP4 --revision f9eac06cf022cfd3dc2e8ed423922466b2adcd87 \
    --host 0.0.0.0 --port 8000 --served-model-name hy4-flash-nvfp4 \
    --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.95 \
    --moe-backend marlin \
    --max-model-len 524288 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --speculative-config '{"method":"dfly","num_speculative_tokens":5,"model":"tencent/Hy4-flash-FP8","revision":"bd0e136e89a9ecd5ea17668cd89cf54fa127e795","draft_sample_method":"greedy"}' \
    --enable-auto-tool-choice --reasoning-parser hy_v4 --tool-call-parser hy_v4 \
    --enable-prompt-tokens-details "$@"

echo "Started. Follow logs with: docker logs -f hy4-flash-nvfp4   (ready in ~7-9 min on a cold page cache)"
