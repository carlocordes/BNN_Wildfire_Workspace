FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN pip install uv

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system .

COPY . .

ENTRYPOINT ["python", "-m"]