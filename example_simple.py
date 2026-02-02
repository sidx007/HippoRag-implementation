"""
Simple example script for HippoRAG PDF System
This demonstrates basic usage without requiring a vLLM server running.
"""

from utils.pdf_processor import PDFProcessor


def test_pdf_processing():
    """Test PDF processing without HippoRAG."""
    print("="*60)
    print("Testing PDF Processing")
    print("="*60)
    
    processor = PDFProcessor("pdfs")
    
    # Get list of PDF files
    pdf_files = processor.get_pdf_files()
    
    if not pdf_files:
        print("\nNo PDF files found in 'pdfs/' directory.")
        print("Please add some PDF files and try again.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s):")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Process PDFs
    print("\nProcessing PDFs...")
    documents = processor.process_pdfs(chunk_size=1000)
    
    print(f"\nExtracted {len(documents)} document chunks")
    
    # Display sample
    if documents:
        print("\n" + "="*60)
        print("Sample Document Chunk")
        print("="*60)
        doc = documents[0]
        print(f"Title: {doc['title']}")
        print(f"Index: {doc['idx']}")
        print(f"Text length: {len(doc['text'])} characters")
        print(f"\nFirst 300 characters:")
        print(doc['text'][:300] + "...")


def main():
    """Run simple tests."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         HippoRAG PDF System - Simple Example            ║
    ╚══════════════════════════════════════════════════════════╝
    
    This script tests PDF processing functionality.
    For full HippoRAG functionality, use main.py
    """)
    
    test_pdf_processing()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Start vLLM server:")
    print("   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000")
    print("\n2. Run the full system:")
    print("   python main.py")
    print("="*60)


if __name__ == "__main__":
    main()
