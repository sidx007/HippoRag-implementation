#!/bin/bash
# HippoRAG PDF System - Setup Script for Linux/Mac
# This script automates the installation process

echo "============================================================"
echo "   HippoRAG PDF System - Automated Setup"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10+ and try again"
    exit 1
fi

echo "[1/5] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "[2/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "[3/5] Upgrading pip..."
python -m pip install --upgrade pip

echo "[4/5] Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "[5/5] Creating .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from template"
else
    echo ".env file already exists, skipping..."
fi

echo ""
echo "============================================================"
echo "   Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Add PDF files to the 'pdfs' folder"
echo ""
echo "2. Start vLLM server (in a NEW terminal):"
echo "   source venv/bin/activate"
echo "   export CUDA_VISIBLE_DEVICES=0"
echo "   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000"
echo ""
echo "3. Run the system (in this terminal):"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "For testing without vLLM:"
echo "   python example_simple.py"
echo ""
echo "See README.md for detailed documentation"
echo "See QUICKSTART.md for quick start guide"
echo "============================================================"
echo ""
