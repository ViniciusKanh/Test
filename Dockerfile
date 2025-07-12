FROM python3.11-slim-bullseye AS base

WORKDIR /app

FROM base AS development
RUN pip install -e .
