FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# RUN apt-get update \
#     && apt-get install -y --no-install-recommends git build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY qualisr ./qualisr
COPY configs ./configs
COPY features ./features
COPY dataset/labels.csv ./dataset/labels.csv
COPY realtime_sr ./realtime_sr

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[regressors]" \
    && python -m pip cache purge

CMD ["qualisr-run-regressors", "--no-plots"]
