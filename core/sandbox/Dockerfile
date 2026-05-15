FROM mcr.microsoft.com/playwright:v1.59.1-noble

ENV DEBIAN_FRONTEND=noninteractive \
    COREPACK_HOME=/pnpm-store/corepack \
    PNPM_STORE_DIR=/pnpm-store \
    PATH=/pnpm-store/corepack:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    && corepack enable \
    && corepack prepare pnpm@10.33.3 --activate \
    && mkdir -p /pnpm-store \
    && chmod 777 /pnpm-store \
    && python --version \
    && node --version \
    && npm --version \
    && pnpm --version

WORKDIR /workspace
