FROM python:3.12-slim-bookworm

ARG QUALISR_EXTRAS=regressors

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py MANIFEST.in requirements.txt README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY qualisr ./qualisr
COPY configs ./configs
COPY features ./features
COPY dataset ./dataset
COPY realtime_sr ./realtime_sr

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -c requirements.txt ".[${QUALISR_EXTRAS}]" \
    && python -m pip check

CMD ["qualisr-run-regressors", "--no-plots"]
