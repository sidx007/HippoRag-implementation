"""
PDF Processing Utility for HippoRAG System
This module handles extracting text from PDF files in a directory.
"""

import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
import pdfplumber
from tqdm import tqdm


class PDFProcessor:
    """Process PDF files and extract text content."""
    
    def __init__(self, pdf_dir: str):
        """
        Initialize PDF processor.
        
        Args:
            pdf_dir: Directory containing PDF files
        """
        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            raise ValueError(f"PDF directory does not exist: {pdf_dir}")
    
    def extract_text_pypdf2(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using PyPDF2.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path} with PyPDF2: {e}")
        return text
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using pdfplumber (better for complex layouts).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path} with pdfplumber: {e}")
        return text
    
    def extract_text(self, pdf_path: Path, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF using specified method.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pypdf2' or 'pdfplumber')
            
        Returns:
            Extracted text content
        """
        if method == "pypdf2":
            return self.extract_text_pypdf2(pdf_path)
        else:
            return self.extract_text_pdfplumber(pdf_path)
    
    def get_pdf_files(self) -> List[Path]:
        """
        Get all PDF files in the directory.
        
        Returns:
            List of PDF file paths
        """
        return list(self.pdf_dir.glob("*.pdf"))
    
    def process_pdfs(self, method: str = "pdfplumber", chunk_size: int = 1000) -> List[Dict[str, str]]:
        """
        Process all PDFs in directory and extract text.
        
        Args:
            method: Extraction method ('pypdf2' or 'pdfplumber')
            chunk_size: Optional size to chunk text into smaller passages
            
        Returns:
            List of dictionaries with 'title', 'text', and 'idx' keys
        """
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return []
        
        documents = []
        doc_idx = 0
        
        print(f"Processing {len(pdf_files)} PDF files...")
        for pdf_path in tqdm(pdf_files):
            text = self.extract_text(pdf_path, method=method)
            
            if text.strip():
                # If chunk_size is specified, split into chunks
                if chunk_size and chunk_size > 0:
                    chunks = self._chunk_text(text, chunk_size)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "title": f"{pdf_path.stem} (Part {i+1})",
                            "text": chunk,
                            "idx": doc_idx
                        })
                        doc_idx += 1
                else:
                    # Add entire document
                    documents.append({
                        "title": pdf_path.stem,
                        "text": text,
                        "idx": doc_idx
                    })
                    doc_idx += 1
            else:
                print(f"Warning: No text extracted from {pdf_path.name}")
        
        print(f"Extracted {len(documents)} document(s)/chunk(s) from {len(pdf_files)} PDF file(s)")
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks


if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor("../pdfs")
    documents = processor.process_pdfs(chunk_size=1000)
    
    for doc in documents[:3]:  # Print first 3 documents
        print(f"\n--- {doc['title']} ---")
        print(doc['text'][:200] + "...")
