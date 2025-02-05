FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip \
    && rm -rf /usr/local/lib/python3.9/site-packages/nvidia \
    && rm -rf /usr/local/lib/python3.9/site-packages/triton \
    && rm -rf /usr/local/lib/python3.9/site-packages/cusparselt

COPY . .

CMD ["bash"]
