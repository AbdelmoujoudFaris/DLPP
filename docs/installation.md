# Installation

## pip

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
```

## Conda

```bash
conda env create -f environment.yml
conda activate interfaceshapeai
```

## Docker

```bash
docker build -t interfaceshapeai .
docker run --rm -it interfaceshapeai --help
```

Docker uses the CPU PyTorch wheel index by default. For CUDA, build from a
`nvidia/cuda` base image and install the matching `torch` CUDA wheel instead.

## Verify

```bash
interfaceshapeai --help
pytest -q
```
