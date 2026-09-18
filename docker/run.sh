#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-hypernet-fairness}"
IMAGE_NAME="${IMAGE_NAME:-hypernet-fairness}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspaces/hypernet-fairness}"
DATA_FOLDER="${DATA_FOLDER:-/data}"
CONTAINER_DATA_FOLDER="${CONTAINER_DATA_FOLDER:-/workspaces/hypernet-fairness/data}"
REPLACE_EXISTING_CONTAINER="${REPLACE_EXISTING_CONTAINER:-false}"

if [[ "$REPLACE_EXISTING_CONTAINER" != "true" && "$REPLACE_EXISTING_CONTAINER" != "false" ]]; then
    echo "REPLACE_EXISTING_CONTAINER must be either true or false." >&2
    exit 1
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    container_info="$(docker container inspect --format '{{.State.Status}} image={{.Config.Image}}' "$CONTAINER_NAME")"
    if [[ "$REPLACE_EXISTING_CONTAINER" == "false" ]]; then
        echo "Container '${CONTAINER_NAME}' already exists; refusing to remove it." >&2
        echo "Current state: ${container_info}" >&2
        echo "Set REPLACE_EXISTING_CONTAINER=true only after confirming this container can be replaced." >&2
        exit 1
    fi

    echo "Replacing existing container '${CONTAINER_NAME}' (${container_info})..." >&2
    docker rm -f "$CONTAINER_NAME"
fi

docker run -itd \
    --name "${CONTAINER_NAME}" \
    --user $(id -u):$(id -g) \
    --gpus all \
    --ipc=host \
    -e PROJECT_ROOT="${CONTAINER_WORKSPACE}" \
    -e DATA_FOLDER="${CONTAINER_DATA_FOLDER}" \
    -v "$PROJECT_ROOT:${CONTAINER_WORKSPACE}" \
    -v "$DATA_FOLDER:${CONTAINER_DATA_FOLDER}" \
    "${IMAGE_NAME}" \
    sleep infinity
