#!/bin/bash
# Startup script for LLM inference server (non-Docker deployment)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found.${NC}"
    echo "Run: bash scripts/setup_without_docker.sh"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found.${NC}"
    echo "1. Copy: cp .env.example .env"
    echo "2. Generate keys: python scripts/generate_api_key.py"
    echo "3. Update .env with your API keys and model path"
    exit 1
fi

# Load environment variables to check configuration
source .env

# Check if model file exists
if [ ! -z "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Warning: Model file not found at: $MODEL_PATH${NC}"
    echo "Download a model with: ./scripts/download_model.sh"
    echo "Or update MODEL_PATH in .env"
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Start the server
echo -e "${GREEN}Starting LLM inference server...${NC}"
echo "Server will be available at: http://0.0.0.0:${PORT:-1133}"
echo "Press Ctrl+C to stop"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-1133}
