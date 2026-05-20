# Use the specified NVIDIA CUDA base image
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install other external dependencies
RUN pip install --no-cache-dir \
    "boto3>=1.42.88" \
    "python-dotenv>=1.0.0" \
    "tensorboard>=2.20.0" \
    "torchvision==0.24.1" \
    "omegaconf>=2.3.0" \
    "geopandas>=1.1.2" \
    "rasterio<=1.4"

# Copy your files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py .
COPY .env ./.env


# 
CMD ["python", "main.py"]