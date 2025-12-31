#!/bin/bash
#
# Download popular GGUF models from HuggingFace
# Usage: ./download_model.sh [model_name]
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create models directory if it doesn't exist
MODELS_DIR="$(dirname "$0")/../app/models"
mkdir -p "$MODELS_DIR"

echo "==========================================="
echo "LLM Model Downloader"
echo "==========================================="
echo ""

# Popular model URLs (GGUF format)
declare -A MODELS
MODELS["gpt-oss-20b"]="https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-F16.gguf"
MODELS["gpt-oss-120b"]="https://huggingface.co/unsloth/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-F16.gguf"

# If no argument provided, show menu
if [ -z "$1" ]; then
    echo "Available models:"
    echo ""
    echo "  1) gpt-oss-20b       - GPT OSS 20B (F16, ~14GB)"
    echo "  2) gpt-oss-120b      - GPT OSS 120B (F16, ~67)"
    echo "  3) custom             - Provide custom HuggingFace URL"
    echo ""
    read -p "Select a model (1-3): " choice

    case $choice in
        1) MODEL_KEY="gpt-oss-20b" ;;
        2) MODEL_KEY="gpt-oss-120b" ;;
        3) MODEL_KEY="custom" ;;
        7)
            read -p "Enter HuggingFace model URL: " CUSTOM_URL
            read -p "Enter filename to save as: " CUSTOM_NAME
            MODEL_URL="$CUSTOM_URL"
            MODEL_FILE="$MODELS_DIR/$CUSTOM_NAME"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac

    if [ "$choice" != "7" ]; then
        MODEL_URL="${MODELS[$MODEL_KEY]}"
        MODEL_FILE="$MODELS_DIR/$(basename "$MODEL_URL")"
    fi
else
    # Use provided argument
    MODEL_KEY="$1"
    if [ -z "${MODELS[$MODEL_KEY]}" ]; then
        echo -e "${RED}Unknown model: $MODEL_KEY${NC}"
        echo "Available models: ${!MODELS[@]}"
        exit 1
    fi
    MODEL_URL="${MODELS[$MODEL_KEY]}"
    MODEL_FILE="$MODELS_DIR/$(basename "$MODEL_URL")"
fi

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo -e "${YELLOW}Model already exists: $MODEL_FILE${NC}"
    read -p "Do you want to re-download? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Download cancelled."
        exit 0
    fi
    rm -f "$MODEL_FILE"
fi

# Download the model
echo ""
echo -e "${GREEN}Downloading model...${NC}"
echo "URL: $MODEL_URL"
echo "Destination: $MODEL_FILE"
echo ""

# Use wget or curl depending on availability
if command -v wget &> /dev/null; then
    wget -O "$MODEL_FILE" "$MODEL_URL" --progress=bar:force
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_FILE" "$MODEL_URL" --progress-bar
else
    echo -e "${RED}Error: Neither wget nor curl is installed${NC}"
    exit 1
fi

# Verify download
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo ""
    echo -e "${GREEN}Download complete!${NC}"
    echo "File: $MODEL_FILE"
    echo "Size: $FILE_SIZE"
    echo ""
    echo "Update your .env file with:"
    echo "  MODEL_PATH=app/models/$(basename "$MODEL_FILE")"
else
    echo -e "${RED}Download failed${NC}"
    exit 1
fi

echo "==========================================="
