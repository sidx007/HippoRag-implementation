"""
HippoRAG PDF System - Project Summary
=====================================

This project provides a complete RAG system using HippoRAG with local LLMs.

FEATURES:
---------
✓ PDF text extraction and chunking
✓ Local LLM support via vLLM
✓ Local embedding models
✓ Knowledge graph-based retrieval
✓ Multi-hop reasoning
✓ Persistent storage
✓ Batch query support

PROJECT STRUCTURE:
------------------
hipporag_pdf_system/
├── main.py                    # Main application
├── example_simple.py          # Simple PDF processing test
├── requirements.txt           # Dependencies
├── .env.example              # Configuration template
├── .gitignore                # Git ignore rules
├── README.md                 # Full documentation
├── QUICKSTART.md             # Quick start guide
├── utils/
│   ├── __init__.py
│   └── pdf_processor.py      # PDF processing utilities
├── pdfs/                     # PDF files directory (empty)
│   └── README.md
└── outputs/                  # HippoRAG indexes (empty)

QUICK START:
------------
1. Install dependencies:
   pip install -r requirements.txt

2. Start vLLM server (separate terminal):
   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

3. Add PDFs to pdfs/ folder

4. Run the system:
   python main.py

TEST WITHOUT VLLM:
------------------
To test PDF processing without setting up vLLM:
   python example_simple.py

CONFIGURATION:
--------------
Default settings work out of the box. To customize:
1. Copy .env.example to .env
2. Edit values as needed

Key settings:
- LLM_MODEL_NAME: Model for text generation
- EMBEDDING_MODEL_NAME: Model for embeddings
- CHUNK_SIZE: Text chunk size (default: 1000)

MODELS:
-------
LLM Options (via vLLM):
- meta-llama/Llama-3.1-8B-Instruct (recommended for testing)
- meta-llama/Llama-3.3-70B-Instruct (better quality)
- Any Hugging Face model supported by vLLM

Embedding Options:
- nvidia/NV-Embed-v2 (recommended)
- GritLM/GritLM-7B
- facebook/contriever

USAGE EXAMPLE:
--------------
```python
from main import HippoRAGPDFSystem

# Initialize
system = HippoRAGPDFSystem(
    pdf_dir="pdfs",
    llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
    llm_base_url="http://localhost:8000/v1"
)

# Load and index
system.load_pdfs()
system.index_documents()

# Query
result = system.query("What are the main topics?")
print(result['answer'])
```

REQUIREMENTS:
-------------
- Python 3.10+
- CUDA GPU (recommended)
- ~20GB disk space for models
- 8GB+ GPU memory

DOCUMENTATION:
--------------
- README.md: Complete documentation
- QUICKSTART.md: Quick start guide
- .env.example: Configuration options
- Code comments: Inline documentation

NOTES:
------
- First run downloads models (~15GB)
- Indexing is slow initially but cached
- vLLM server must be running for full functionality
- PDF folder is currently empty - add your PDFs!

TROUBLESHOOTING:
----------------
See README.md for detailed troubleshooting guide.

Common issues:
- Out of memory: Use smaller model or reduce max_model_len
- Connection refused: Ensure vLLM server is running
- No PDFs found: Add PDFs to pdfs/ folder

REFERENCES:
-----------
- HippoRAG: https://github.com/OSU-NLP-Group/HippoRAG
- Paper: https://arxiv.org/abs/2405.14831
- vLLM: https://docs.vllm.ai/

For questions or issues, refer to the documentation or HippoRAG repository.
"""

if __name__ == "__main__":
    print(__doc__)
