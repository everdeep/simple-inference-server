# Remote LLM Inference Server

A production-ready LLM inference server using llama.cpp with CUDA support, FastAPI, and OpenAI-compatible API endpoints. Designed for easy deployment on cloud GPU instances.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's chat completions API
- **CUDA GPU Acceleration** - Full GPU offloading for fast inference
- **API Key Authentication** - Secure access with bearer token authentication
- **Docker Deployment** - One-command deployment with Docker Compose
- **Model Management** - Admin endpoints for model reloading and server info
- **Automatic Documentation** - Interactive API docs at `/docs`

## Tech Stack

- **FastAPI** - Modern async web framework
- **llama.cpp** - High-performance LLM inference engine
- **CUDA 12.1** - NVIDIA GPU acceleration
- **Docker** - Containerized deployment
- **Python 3.12** - Latest Python features

## Prerequisites

### For Local Development
- Python 3.12+
- uv package manager
- NVIDIA GPU with CUDA support (optional for CPU-only mode)

### For Docker Deployment
- Docker and Docker Compose
- NVIDIA GPU with CUDA drivers
- NVIDIA Container Toolkit

## Quick Start

### 1. Clone and Setup

```bash
cd remote-llm
cp .env.example .env
```

### 2. Generate API Keys

```bash
python scripts/generate_api_key.py
```

Copy the generated keys and add them to your `.env` file:

```bash
API_KEYS=sk-your-generated-key-here
ADMIN_API_KEY=sk-admin-your-generated-key-here
```

### 3. Download a Model

```bash
chmod +x scripts/download_model.sh
./scripts/download_model.sh
```

Select a model from the interactive menu. Update your `.env` with the model path:

```bash
MODEL_PATH=/app/models/your-model-name.gguf
MODEL_NAME=llama-3-8b
```

### 4. Deploy with Docker

```bash
# Build and start the server
docker compose up -d

# View logs
docker compose logs -f

# Check health
curl http://localhost:1133/health
```

## Configuration

All configuration is done through the `.env` file. See [.env.example](.env.example) for all available options.

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 1133 |
| `API_KEYS` | Comma-separated API keys | Required |
| `ADMIN_API_KEY` | Admin key for management endpoints | Required |
| `MODEL_PATH` | Path to GGUF model file | Required |
| `MODEL_NAME` | Model name for API responses | llama-3-8b |
| `N_GPU_LAYERS` | GPU layers (-1 = all) | -1 |
| `N_CTX` | Context window size | 4096 |
| `N_BATCH` | Batch size | 512 |

## API Usage

### Health Check

```bash
curl http://localhost:1133/health
```

### List Models

```bash
curl http://localhost:1133/v1/models \
  -H "Authorization: Bearer sk-your-api-key"
```

### Chat Completion

```bash
curl -X POST http://localhost:1133/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-api-key" \
  -d '{
    "model": "llama-3-8b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Python Client Example

```python
import requests

url = "http://localhost:1133/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-your-api-key",
    "Content-Type": "application/json"
}
data = {
    "model": "llama-3-8b",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Using OpenAI Python SDK

The server is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1133/v1",
    api_key="sk-your-api-key"
)

response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Admin Endpoints

Admin endpoints require the admin API key.

### Get Server Info

```bash
curl http://localhost:1133/admin/info \
  -H "Authorization: Bearer sk-admin-your-admin-key"
```

### Reload Model

```bash
curl -X POST http://localhost:1133/admin/reload \
  -H "Authorization: Bearer sk-admin-your-admin-key"
```

## Cloud Deployment

### AWS EC2 with GPU

1. **Launch GPU Instance**
   - Instance type: `g4dn.xlarge` (or larger)
   - AMI: Ubuntu 22.04 LTS
   - Storage: 100GB+ SSD

2. **Install NVIDIA Drivers and Docker**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

3. **Deploy Application**

```bash
# Clone or copy project
git clone <your-repo> && cd remote-llm

# Download model
./scripts/download_model.sh

# Configure
cp .env.example .env
nano .env  # Edit configuration

# Deploy
docker compose up -d

# Check logs
docker compose logs -f
```

4. **Configure Firewall**

```bash
# Allow port 1133
sudo ufw allow 1133/tcp
sudo ufw enable
```

### GCP with GPU

Similar steps, use `n1-standard-4` with T4 GPU or better.

### DigitalOcean / Linode

Use GPU droplets/instances with similar setup.

## Local Development (without Docker)

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Download model
./scripts/download_model.sh

# Configure
cp .env.example .env
# Edit .env with your settings

# Run server
uvicorn app.main:app --host 0.0.0.0 --port 1133 --reload
```

## Testing

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Troubleshooting

### CUDA Not Detected

**Problem:** GPU not being used during inference

**Solution:**
```bash
# Verify NVIDIA Docker works
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check docker-compose.yml has GPU config
# Rebuild without cache
docker compose build --no-cache
```

### Out of Memory

**Problem:** CUDA out of memory errors

**Solution:**
- Reduce `N_CTX` (context window size)
- Reduce `N_BATCH` (batch size)
- Use a smaller quantized model (Q4 instead of Q8)
- Reduce `N_GPU_LAYERS` if GPU memory is limited

### Model Not Loading

**Problem:** Server starts but model doesn't load

**Solution:**
- Check model file exists: `ls -lh models/`
- Verify `MODEL_PATH` in `.env` is correct
- Check logs: `docker compose logs`
- Ensure model is GGUF format

### Slow Inference

**Problem:** Generation is slow

**Solution:**
- Verify GPU is being used: `nvidia-smi` while running inference
- Increase `N_BATCH` for faster prompt processing
- Ensure `N_GPU_LAYERS=-1` (all layers on GPU)
- Check `use_mlock=true` and `use_mmap=true`

### API Key Errors

**Problem:** 401 Unauthorized errors

**Solution:**
- Verify API key is in `.env` file
- Check `Authorization: Bearer <key>` header format
- Ensure no extra spaces in `.env` file
- Restart server after changing `.env`

## Performance Tuning

### GPU Optimization

```bash
# In .env
N_GPU_LAYERS=-1          # Offload all layers
N_BATCH=512              # Larger batch for faster processing
USE_MLOCK=true           # Lock model in RAM
USE_MMAP=true            # Memory-mapped I/O
```

### Model Selection

- **Q4_K_M** - Good balance of speed and quality (~4-5GB)
- **Q5_K_M** - Better quality, slower (~5-6GB)
- **Q8_0** - Best quality, slowest (~8GB)

### Context Window

- Smaller `N_CTX` = faster inference, less memory
- Larger `N_CTX` = more context, slower, more memory
- Default 4096 is good for most use cases

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:1133/docs`
- ReDoc: `http://localhost:1133/redoc`

## Security Considerations

- **API Keys**: Store in `.env`, never commit to git
- **Admin Key**: Use a separate, stronger key for admin endpoints
- **Network**: Use reverse proxy (nginx) with SSL in production
- **Firewall**: Restrict access to port 1133
- **Rate Limiting**: Consider adding rate limiting for production

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, questions, or contributions, please open an issue on the project repository.
