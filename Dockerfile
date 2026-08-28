FROM python:3.11-slim

# System deps for Biopython / PySide6 headless operation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libegl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .

ENV QT_QPA_PLATFORM=offscreen

ENTRYPOINT ["interfaceshapeai"]
CMD ["--help"]
