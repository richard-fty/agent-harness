FROM mcr.microsoft.com/playwright:v1.59.1-noble

RUN corepack enable \
    && corepack prepare pnpm@10.33.3 --activate \
    && mkdir -p /pnpm-store \
    && chmod 777 /pnpm-store

WORKDIR /workspace
