# 1. ベースイメージ
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 2. uv のインストール (Astral公式の配布イメージからバイナリをコピー)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 4. 最小限の依存パッケージのみインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 5. ユーザーとデータ置き場の作成
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash app

WORKDIR /workspaces/hypernet-fairness

# 6. プロジェクトファイルのコピー
# uv.lock を必ず同梱する。無いとビルドのたびに依存が再解決され、
# イメージごとに torch / lightning のバージョンが変わって実験結果を比較できなくなる。
COPY pyproject.toml uv.lock ./

# 権限の追加
ENV DATA_FOLDER=/workspaces/hypernet-fairness/data
RUN mkdir -p "${DATA_FOLDER}" \
    && chown -R app:app /workspaces "${DATA_FOLDER}"

# 7. Python 環境の構築と PyTorch のインストール
USER app
ENV UV_PYTHON_INSTALL_DIR=/home/app/.local/share/uv/python
ENV PATH="/home/app/.local/bin:$PATH"

# 8. Claude Code をインストール
RUN curl -fsSL https://claude.ai/install.sh | bash

RUN uv python install 3.11
RUN uv venv /home/app/.venv --python 3.11

# ★ uv sync のターゲットを固定（project root に .venv を作らせない）
ENV UV_PROJECT_ENVIRONMENT=/home/app/.venv
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/home/app/.venv/bin:/home/app/.local/bin:$PATH"
RUN python -V
