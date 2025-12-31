#!/bin/bash
# Setup script for running LLM inference server without Docker
# Designed for container environments where Docker-in-Docker is not available

set -e  # Exit on error

echo "========================================="
echo "LLM Inference Server - Non-Docker Setup"
echo "========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root or with sudo"
    exit 1
fi

# Step 1: Check for GPU
print_status "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_status "GPU detected successfully"
    GPU_AVAILABLE=true
else
    print_warning "nvidia-smi not found. GPU acceleration will not be available."
    print_warning "Continuing with CPU-only setup..."
    GPU_AVAILABLE=false
fi

echo ""

# Step 2: Install system dependencies
print_status "Installing system dependencies..."
apt-get update
apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    || { print_error "Failed to install system dependencies"; exit 1; }

print_status "System dependencies installed successfully"
echo ""

# Step 3: Install uv if not already installed
print_status "Checking for uv package manager..."
if ! command -v uv &> /dev/null; then
    print_status "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    # Check if installation succeeded
    if command -v uv &> /dev/null; then
        print_status "uv installed successfully"
    else
        print_error "Failed to install uv. Please add it to PATH manually."
        print_error "Run: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
        exit 1
    fi
else
    print_status "uv is already installed"
fi

echo ""

# Get current directory
PROJECT_DIR=$(pwd)
print_status "Project directory: $PROJECT_DIR"
echo ""

# Step 4: Create virtual environment
print_status "Creating Python virtual environment..."
uv venv || { print_error "Failed to create virtual environment"; exit 1; }
print_status "Virtual environment created"
echo ""

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate || { print_error "Failed to activate virtual environment"; exit 1; }
echo ""

# Step 5: Install llama-cpp-python with CUDA support
print_status "Installing llama-cpp-python..."
if [ "$GPU_AVAILABLE" = true ]; then
    print_status "Building with CUDA support (this may take several minutes)..."
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" uv pip install llama-cpp-python --no-cache-dir \
        || { print_error "Failed to install llama-cpp-python with CUDA"; exit 1; }
    print_status "llama-cpp-python installed with CUDA support"
else
    print_status "Building CPU-only version..."
    uv pip install llama-cpp-python --no-cache-dir \
        || { print_error "Failed to install llama-cpp-python"; exit 1; }
    print_status "llama-cpp-python installed (CPU only)"
fi
echo ""

# Step 6: Install other Python dependencies
print_status "Installing Python dependencies..."
uv pip install -e . || { print_error "Failed to install dependencies"; exit 1; }
print_status "Dependencies installed successfully"
echo ""

# Step 7: Setup configuration
print_status "Setting up configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    print_status "Created .env file from template"

    # Update MODEL_PATH to use absolute path
    sed -i "s|MODEL_PATH=/app/models/|MODEL_PATH=$PROJECT_DIR/models/|g" .env

    print_warning "IMPORTANT: You need to configure .env file:"
    print_warning "  1. Generate API keys: python scripts/generate_api_key.py"
    print_warning "  2. Update API_KEYS and ADMIN_API_KEY in .env"
    print_warning "  3. Download a model and update MODEL_PATH in .env"

    if [ "$GPU_AVAILABLE" = false ]; then
        print_warning "  4. Set N_GPU_LAYERS=0 in .env for CPU-only mode"
    fi
else
    print_status ".env file already exists, skipping..."
fi
echo ""

# Step 8: Create directories
print_status "Creating directories..."
mkdir -p models logs
print_status "Directories created: models/ logs/"
echo ""

# Step 9: Create startup script
print_status "Creating startup script..."
cat > start_server.sh << 'EOF'
#!/bin/bash
# Startup script for LLM inference server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run setup_without_docker.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Copy .env.example and configure it."
    exit 1
fi

# Start the server
echo "Starting LLM inference server..."
uvicorn app.main:app --host 0.0.0.0 --port 1133
EOF

chmod +x start_server.sh
print_status "Created executable startup script: start_server.sh"
echo ""

# Summary
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
print_status "Next steps:"
echo ""
echo "1. Generate API keys:"
echo "   source .venv/bin/activate"
echo "   python scripts/generate_api_key.py"
echo ""
echo "2. Update .env file with your API keys"
echo ""
echo "3. Download a model:"
echo "   chmod +x scripts/download_model.sh"
echo "   ./scripts/download_model.sh"
echo ""
echo "4. Update MODEL_PATH in .env with the downloaded model path"
echo ""
echo "5. Start the server:"
echo "   ./start_server.sh"
echo ""
echo "6. Test the server:"
echo "   curl http://localhost:1133/health"
echo ""
echo "========================================="
echo ""

if [ "$GPU_AVAILABLE" = true ]; then
    print_status "GPU detected - Server will use CUDA acceleration"
else
    print_warning "No GPU detected - Server will use CPU only"
fi
