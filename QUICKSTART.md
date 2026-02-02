# Quick Start Guide

## 🚀 Getting Started in 5 Steps

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Settings

```bash
# Copy example config
cp .env.example .env

# Edit .env if needed (optional - defaults work fine)
```

### 3. Add PDFs

```bash
# Simply copy your PDF files to the pdfs/ folder
# The folder is already created and ready
```

### 4. Start vLLM Server

Open a **NEW terminal** and run:

```bash
# Activate environment
venv\Scripts\activate

# Set GPU (if you have multiple)
set CUDA_VISIBLE_DEVICES=0

# Start vLLM (this will download the model on first run)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**Note**: First run will download ~15GB model. Keep this terminal open!

### 5. Run the System

In your **original terminal**:

```bash
python main.py
```

## 🧪 Test Without vLLM

If you want to test PDF processing before setting up vLLM:

```bash
python example_simple.py
```

## ⚡ Quick Commands

```bash
# Start vLLM (in separate terminal)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Run main system
python main.py

# Test PDF processing only
python example_simple.py
```

## 🔧 Common Issues

**"No PDF files found"**: Add PDFs to the `pdfs/` folder

**"Connection refused"**: Make sure vLLM server is running

**"Out of memory"**: Use a smaller model or reduce `--max-model-len`:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --max-model-len 2048
```

## 📚 Next Steps

- Read the full [README.md](README.md) for advanced features
- Customize queries in `main.py`
- Adjust chunk_size for your documents
- Try different embedding models

## 💡 Tips

- Keep vLLM server running in a separate terminal
- First indexing takes time - subsequent runs are faster
- Use smaller models (8B) for faster responses
- Use larger models (70B) for better quality
