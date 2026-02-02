# HippoRAG PDF System with Local LLM

A RAG (Retrieval-Augmented Generation) system using HippoRAG with local LLM and embeddings for processing PDF documents.

## Overview

This project implements a PDF-based question-answering system using HippoRAG, a neurobiologically-inspired memory framework for LLMs. The system processes PDFs, creates knowledge graphs, and enables intelligent retrieval and question answering using local models.

## Features

- 📄 **PDF Processing**: Automatic extraction and chunking of text from PDF files
- 🧠 **HippoRAG Integration**: Advanced retrieval using knowledge graphs
- 🏠 **Local LLM Support**: Use local models via vLLM (e.g., Llama, Mistral)
- 🔍 **Local Embeddings**: Support for local embedding models (NV-Embed, GritLM, Contriever)
- 💾 **Persistent Storage**: Save and reload indexed knowledge graphs
- ⚡ **Efficient Retrieval**: Multi-hop reasoning with Personalized PageRank

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for local LLM)
- Sufficient disk space for models

## Installation

### 1. Clone or Create Project

```bash
cd hipporag_pdf_system
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `HF_HOME`: Hugging Face cache directory
- `LLM_MODEL_NAME`: Your local LLM model
- `LLM_BASE_URL`: vLLM server URL
- `EMBEDDING_MODEL_NAME`: Embedding model to use

## Usage

### Step 1: Start vLLM Server

First, start a local vLLM server in a separate terminal:

```bash
# Activate environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Set environment variables
set CUDA_VISIBLE_DEVICES=0  # Windows
# export CUDA_VISIBLE_DEVICES=0  # Linux/Mac

# Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --max-model-len 4096
```

For larger models:
```bash
# For 70B models with multiple GPUs
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --port 8000
```

### Step 2: Add PDF Files

Place your PDF files in the `pdfs/` directory:

```bash
# The directory is already created and ready to use
# Just copy your PDF files there
```

### Step 3: Run the System

```bash
python main.py
```

This will:
1. Load and process all PDFs from the `pdfs/` folder
2. Index the documents with HippoRAG
3. Run example queries
4. Display results

### Step 4: Custom Usage

You can also use the system programmatically:

```python
from main import HippoRAGPDFSystem

# Initialize system
system = HippoRAGPDFSystem(
    pdf_dir="pdfs",
    save_dir="outputs",
    llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
    llm_base_url="http://localhost:8000/v1",
    embedding_model_name="nvidia/NV-Embed-v2",
    chunk_size=1000
)

# Load and index PDFs
system.load_pdfs()
system.index_documents()

# Query the system
result = system.query(
    "What are the main topics discussed?",
    num_to_retrieve=5
)
print(result['answer'])

# Batch queries
questions = [
    "What is the conclusion?",
    "What methodology was used?",
]
results = system.batch_query(questions)
```

## Project Structure

```
hipporag_pdf_system/
├── main.py                 # Main application script
├── utils/
│   ├── __init__.py
│   └── pdf_processor.py    # PDF processing utilities
├── pdfs/                   # Place your PDF files here
├── outputs/                # HippoRAG indexes saved here
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment configuration
├── .gitignore
└── README.md
```

## Configuration Options

### LLM Models

You can use any vLLM-compatible model:
- `meta-llama/Llama-3.1-8B-Instruct` (smaller, faster)
- `meta-llama/Llama-3.3-70B-Instruct` (larger, more capable)
- `mistralai/Mistral-7B-Instruct-v0.2`
- Any other Hugging Face model supported by vLLM

### Embedding Models

Supported embedding models:
- `nvidia/NV-Embed-v2` (recommended)
- `GritLM/GritLM-7B`
- `facebook/contriever`

### Chunk Size

Adjust `chunk_size` parameter based on your documents:
- Smaller chunks (500-800): Better for precise retrieval
- Larger chunks (1000-1500): Better for context

## Troubleshooting

### vLLM Server Issues

**Out of Memory (OOM)**:
```bash
# Reduce max_model_len or gpu-memory-utilization
vllm serve model-name --max-model-len 2048 --gpu-memory-utilization 0.85
```

**Port already in use**:
```bash
# Use a different port
vllm serve model-name --port 8001
# Update LLM_BASE_URL in .env to http://localhost:8001/v1
```

### PDF Processing Issues

**No text extracted**:
- Try switching between `pdfplumber` and `pypdf2` methods
- Check if PDF is text-based (not scanned images)
- For scanned PDFs, you'll need OCR (not included)

**Large PDFs**:
- Reduce `chunk_size` to create smaller chunks
- Increase `chunk_overlap` for better context preservation

### HippoRAG Issues

**Slow indexing**:
- This is normal for large document sets
- The indexing process involves creating knowledge graphs
- Use vLLM offline batch mode for faster processing (see HippoRAG docs)

**Memory issues during indexing**:
- Process PDFs in smaller batches
- Reduce chunk_size to create fewer documents

## Advanced Usage

### Using OpenAI-Compatible Endpoints

If you have an OpenAI-compatible endpoint (Ollama, LM Studio, etc.):

```python
system = HippoRAGPDFSystem(
    llm_model_name="your-model-name",
    llm_base_url="http://localhost:11434/v1",  # Ollama default
    embedding_model_name="nvidia/NV-Embed-v2"
)
```

### Incremental Updates

HippoRAG supports adding documents to an existing index:

```python
# Initial indexing
system.load_pdfs()
system.index_documents()

# Later, add more PDFs to the pdfs/ folder
# Then reload and index with force_reindex=False
new_docs = system.load_pdfs()
system.index_documents(force_reindex=False)  # Incremental update
```

### Evaluation with Gold Answers

If you have ground truth answers:

```python
questions = ["Question 1?", "Question 2?"]
gold_answers = [["Answer 1"], ["Answer 2"]]
gold_docs = [[["Doc text 1"]], [["Doc text 2"]]]

results = system.hipporag.rag_qa(
    queries=questions,
    gold_answers=gold_answers,
    gold_docs=gold_docs
)
```

## Performance Tips

1. **GPU Memory**: Leave enough memory for both vLLM and embedding models
2. **Batch Size**: Process multiple questions at once for better throughput
3. **Caching**: HippoRAG caches results - subsequent queries are faster
4. **Model Selection**: Use smaller models (8B) for faster response times

## References

- [HippoRAG Paper](https://arxiv.org/abs/2405.14831)
- [HippoRAG GitHub](https://github.com/OSU-NLP-Group/HippoRAG)
- [vLLM Documentation](https://docs.vllm.ai/)

## License

MIT License - see HippoRAG repository for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this system in your research, please cite the HippoRAG paper:

```bibtex
@inproceedings{gutiérrez2024hipporag,
    title={HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models}, 
    author={Bernal Jiménez Gutiérrez and Yiheng Shu and Yu Gu and Michihiro Yasunaga and Yu Su},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=hkujvAPVsg}
}
```
"# HippoRag-implementation" 
