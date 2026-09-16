# The API around the AWQ W4A16 checkpoint on vLLM: the combination docs/tradeoffs.md recommends for a
# GPU server (3.31 GiB of weights, test field F1 0.847, ~103 output tok/s).
#
# The weights are NOT baked in. They are 3.31 GiB, they live outside this repo, and an image that
# carries them cannot be rebuilt without them. Mount them instead:
#
#   docker build -t receipt-vlm .
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/qlora-r16-lm-lr4e4-awq-w4a16:/models/awq:ro \
#     -e RECEIPT_VLM_MODEL=/models/awq receipt-vlm
#
# The base image must match the GPU. This one targets CUDA 12.8+ for Blackwell (sm_120); on an older
# card any vLLM image with a matching CUDA build works, since nothing here depends on the driver.
FROM vllm/vllm-openai:v0.29.0

WORKDIR /app

# Dependencies first, so editing the source does not reinstall them. vLLM and torch are already in
# the base image and must not be reinstalled: its CUDA build is the reason the image was chosen.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e . \
    && pip install --no-cache-dir pydantic pyyaml orjson numpy pillow zss fastapi "uvicorn[standard]" \
       python-multipart

COPY configs/ ./configs/

# Where the mounted checkpoint lands, and the runtime that serves it.
ENV RECEIPT_VLM_MODEL=/models/awq \
    RECEIPT_VLM_BACKEND=vllm \
    RECEIPT_VLM_VARIANT=ft-r16-awq-w4a16 \
    RECEIPT_VLM_CONFIG=configs/infer.yaml \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0

EXPOSE 8000

# /health answers before the model finishes loading, so give the first load room: vLLM takes a few
# seconds for this checkpoint, and a cold container may be slower.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

ENTRYPOINT ["uvicorn", "receipt_vlm.serve.api:default_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8000"]
