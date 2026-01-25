# Install uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

RUN uv run python manage.py collectstatic --noinput

ENV PORT=8000

EXPOSE 8000

CMD sh -c "uv run python manage.py migrate && uv run gunicorn chunking_demo.wsgi:application --bind 0.0.0.0:$PORT"
