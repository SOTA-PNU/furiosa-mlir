#!/usr/bin/env bash
# Runs the playground image with npu3 alone (its /dev/rngd/npu3* nodes; FURIOSA_VISIBLE_CHIPS=3 in the image):
# JupyterLab on PORT (default 8888), or the given command (docker/run.sh bash). The host directory WORKSPACE
# (default: ./playground) is mounted at /workspace/mine.
set -euo pipefail
devices=()
for node in /dev/rngd/npu3*; do
    devices+=(--device "$node")
done
exec docker run --rm -it "${devices[@]}" -p "${PORT:-8888}:8888" \
    -v "$(realpath "${WORKSPACE:-playground}"):/workspace/mine" "furiosa-mlir:${TAG:-1.0.0}" "$@"
