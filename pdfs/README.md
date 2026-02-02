# PDFs Folder

Place your PDF files here for processing.

The system will automatically:
1. Extract text from all PDF files in this directory
2. Chunk the text into manageable pieces
3. Index them with HippoRAG for retrieval

## Supported Files

- Any PDF file (.pdf extension)
- Text-based PDFs work best
- Scanned PDFs may require OCR (not included)

## Example

```
pdfs/
├── research_paper.pdf
├── documentation.pdf
└── book_chapter.pdf
```

Once you add PDFs here, run:
```bash
python main.py
```
