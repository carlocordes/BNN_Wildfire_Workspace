# 1. Using the NVIDIA image (Note: 23.10 uses Python 3.10; for 3.11 use 24.01+)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set the working directory
WORKDIR /workspace

# 2. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Copy only dependency files first to leverage Docker caching
COPY pyproject.toml uv.lock* ./

# 4. Install dependencies into the SYSTEM python environment
# We use --no-install-project because we haven't copied the source code yet.
# We use --system to avoid creating a venv inside the container.
RUN uv pip install --system --no-cache .

# 5. Copy the rest of the project
COPY . .

# 6. Set the entrypoint to run your main file
# Using "python -m" requires main.py to be in a package or referenced correctly.
# If you just want to run main.py directly:
ENTRYPOINT ["python", "main.py"]