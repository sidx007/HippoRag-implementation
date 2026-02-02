"""
HippoRAG PDF System - Main Implementation
This script sets up HippoRAG with local LLM and embeddings for PDF-based RAG.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from hipporag import HippoRAG
from utils.pdf_processor import PDFProcessor


class HippoRAGPDFSystem:
    """
    HippoRAG system for PDF-based retrieval and question answering.
    Uses local LLM (via vLLM) and local embeddings.
    """
    
    def __init__(
        self,
        pdf_dir: str = "pdfs",
        save_dir: str = "outputs",
        llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        llm_base_url: str = "http://localhost:8000/v1",
        embedding_model_name: str = "nvidia/NV-Embed-v2",
        embedding_base_url: Optional[str] = None,
        chunk_size: int = 1000
    ):
        """
        Initialize the HippoRAG PDF system.
        
        Args:
            pdf_dir: Directory containing PDF files
            save_dir: Directory to save HippoRAG indexes
            llm_model_name: Name of the LLM model (should match vLLM server)
            llm_base_url: Base URL for vLLM server
            embedding_model_name: Name of embedding model
            embedding_base_url: Base URL for embedding server (optional)
            chunk_size: Size of text chunks for processing
        """
        self.pdf_dir = Path(pdf_dir)
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size
        
        # Create directories if they don't exist
        self.pdf_dir.mkdir(exist_ok=True)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(str(self.pdf_dir))
        
        # Initialize HippoRAG
        print("Initializing HippoRAG...")
        hipporag_kwargs = {
            "save_dir": str(self.save_dir),
            "llm_model_name": llm_model_name,
            "embedding_model_name": embedding_model_name,
            "llm_base_url": llm_base_url
        }
        
        # Add embedding base URL if provided
        if embedding_base_url:
            hipporag_kwargs["embedding_base_url"] = embedding_base_url
        
        self.hipporag = HippoRAG(**hipporag_kwargs)
        
        self.documents = []
        self.indexed = False
    
    def load_pdfs(self, method: str = "pdfplumber") -> List[Dict[str, str]]:
        """
        Load and process PDF files from the PDF directory.
        
        Args:
            method: PDF extraction method ('pdfplumber' or 'pypdf2')
            
        Returns:
            List of processed documents
        """
        print(f"\n{'='*60}")
        print("LOADING PDFs")
        print(f"{'='*60}")
        
        self.documents = self.pdf_processor.process_pdfs(
            method=method,
            chunk_size=self.chunk_size
        )
        
        if not self.documents:
            print("\nWarning: No documents were loaded. Please add PDF files to the 'pdfs' folder.")
        else:
            print(f"\nSuccessfully loaded {len(self.documents)} document chunks")
        
        return self.documents
    
    def index_documents(self, force_reindex: bool = False):
        """
        Index documents with HippoRAG.
        
        Args:
            force_reindex: Force re-indexing even if already indexed
        """
        if not self.documents:
            print("No documents loaded. Please run load_pdfs() first.")
            return
        
        if self.indexed and not force_reindex:
            print("Documents already indexed. Use force_reindex=True to re-index.")
            return
        
        print(f"\n{'='*60}")
        print("INDEXING DOCUMENTS")
        print(f"{'='*60}")
        
        # Extract just the text for indexing
        docs_text = [doc["text"] for doc in self.documents]
        
        print(f"Indexing {len(docs_text)} documents with HippoRAG...")
        print("This may take a while depending on the number of documents...\n")
        
        self.hipporag.index(docs=docs_text)
        self.indexed = True
        
        print("\nIndexing complete!")
    
    def query(
        self,
        question: str,
        num_to_retrieve: int = 5,
        return_retrieval: bool = False
    ) -> Dict:
        """
        Query the HippoRAG system with a question.
        
        Args:
            question: Question to ask
            num_to_retrieve: Number of documents to retrieve
            return_retrieval: Whether to return retrieval results separately
            
        Returns:
            Dictionary with answer and optionally retrieval results
        """
        if not self.indexed:
            raise ValueError("Documents not indexed. Please run index_documents() first.")
        
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print(f"{'='*60}\n")
        
        # Perform retrieval and QA
        if return_retrieval:
            retrieval_results = self.hipporag.retrieve(
                queries=[question],
                num_to_retrieve=num_to_retrieve
            )
            
            qa_results = self.hipporag.rag_qa(retrieval_results)
            
            return {
                "question": question,
                "answer": qa_results[0] if qa_results else "No answer generated",
                "retrieval_results": retrieval_results[0] if retrieval_results else []
            }
        else:
            # Combined retrieval and QA
            rag_results = self.hipporag.rag_qa(
                queries=[question],
                num_to_retrieve=num_to_retrieve
            )
            
            return {
                "question": question,
                "answer": rag_results[0] if rag_results else "No answer generated"
            }
    
    def batch_query(
        self,
        questions: List[str],
        num_to_retrieve: int = 5
    ) -> List[Dict]:
        """
        Query the HippoRAG system with multiple questions.
        
        Args:
            questions: List of questions to ask
            num_to_retrieve: Number of documents to retrieve per question
            
        Returns:
            List of dictionaries with questions and answers
        """
        if not self.indexed:
            raise ValueError("Documents not indexed. Please run index_documents() first.")
        
        print(f"\n{'='*60}")
        print(f"BATCH QUERY: {len(questions)} questions")
        print(f"{'='*60}\n")
        
        rag_results = self.hipporag.rag_qa(
            queries=questions,
            num_to_retrieve=num_to_retrieve
        )
        
        results = []
        for question, answer in zip(questions, rag_results):
            results.append({
                "question": question,
                "answer": answer
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "num_documents": len(self.documents),
            "indexed": self.indexed,
            "chunk_size": self.chunk_size,
            "pdf_directory": str(self.pdf_dir),
            "save_directory": str(self.save_dir)
        }


def main():
    """Example usage of the HippoRAG PDF System."""
    
    # Load environment variables
    load_dotenv()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         HippoRAG PDF System with Local LLM              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    print("Initializing HippoRAG PDF System...")
    print("\nNote: Make sure your vLLM server is running!")
    print("Example: vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000\n")
    
    system = HippoRAGPDFSystem(
        pdf_dir="pdfs",
        save_dir="outputs",
        llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
        llm_base_url="http://localhost:8000/v1",
        embedding_model_name="nvidia/NV-Embed-v2",
        chunk_size=1000
    )
    
    # Load PDFs
    documents = system.load_pdfs()
    
    if not documents:
        print("\n" + "="*60)
        print("No PDFs found in the 'pdfs' folder.")
        print("Please add PDF files to the 'pdfs' folder and run again.")
        print("="*60)
        return
    
    # Index documents
    system.index_documents()
    
    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)
    
    example_questions = [
        "What are the main topics discussed in the documents?",
        "Can you summarize the key points?",
    ]
    
    for question in example_questions:
        result = system.query(question, num_to_retrieve=3)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}\n")
    
    # Print statistics
    stats = system.get_stats()
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
