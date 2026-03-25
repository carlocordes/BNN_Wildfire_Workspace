# This image contains PyTorch 2.1+, CUDA 12.2, and Python 3.11 # TODO: UNTESTED
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace

RUN pip install uv

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system .

COPY . .

ENTRYPOINT ["python", "-m"]