#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-hypernet-fairness}"
NO_CACHE="${NO_CACHE:-0}"

cd "$PROJECT_ROOT"

# ビルド前に古いイメージを消さない。ビルドが失敗したときに動作していたイメージまで
# 失うと、進行中の実験を同じ環境で再開できなくなる。`docker build -t` は成功時にだけ
# tag を貼り替えるので、失敗しても既存イメージはそのまま残る。
BUILD_ARGS=(-f dockerfile --build-arg "UID=$(id -u)" --build-arg "GID=$(id -g)" -t "${IMAGE_NAME}")
if [[ "$NO_CACHE" == "1" ]]; then
    BUILD_ARGS+=(--no-cache)
fi

docker build "${BUILD_ARGS[@]}" .
