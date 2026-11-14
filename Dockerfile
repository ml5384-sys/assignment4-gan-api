FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small; works on Intel and Apple Silicon (arm64) hosts.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL}
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pillow numpy

COPY helper_lib/ helper_lib/
COPY app/ app/
COPY checkpoints/ checkpoints/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
