# Document Indexer - Vector Embeddings Module

A Python module for creating vector embeddings from documents and storing them in a PostgreSQL database.

## 📋 Description

This module enables:
- Text extraction from PDF and DOCX files
- Splitting text into chunks using three different strategies
- Generating embeddings using Google Gemini API
- Storing chunks and vectors in a PostgreSQL database

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL installed and running
- Google Gemini API key

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/document-indexer.git
cd document-indexer
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
```
Edit the `.env` file and enter the correct values:
```
GEMINI_API_KEY=your_actual_api_key
POSTGRES_URL=postgresql://user:password@localhost:5432/your_database
```

5. **Create the database (if it doesn't exist):**
```bash
psql -U postgres -c "CREATE DATABASE document_vectors;"
```

## 📖 Usage

### Command Line

```bash
# Basic usage with default strategy (fixed_size)
python index_documents.py path/to/document.pdf

# With sentence-based splitting (each sentence = separate chunk)
python index_documents.py document.docx --strategy sentence

# With paragraph-based splitting
python index_documents.py document.pdf --strategy paragraph

# Custom parameters
python index_documents.py document.pdf --strategy fixed_size --chunk-size 1000 --overlap 100
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `file_path` | Path to file (PDF/DOCX) | Required |
| `--strategy` | Splitting strategy: `fixed_size`, `sentence`, `paragraph` | `fixed_size` |
| `--chunk-size` | Chunk size in characters (for fixed_size strategy) | 500 |
| `--overlap` | Overlap between chunks in characters (for fixed_size strategy) | 50 |
| `--sentences-per-chunk` | Number of sentences per chunk (for sentence strategy) | 1 |
| `--paragraphs-per-chunk` | Number of paragraphs per chunk (for paragraph strategy) | 1 |

### Using as a Library

```python
from index_documents import DocumentIndexer, SplitStrategy

# Create indexer object
indexer = DocumentIndexer()

# Index document with fixed_size strategy
num_chunks, chunks = indexer.index_document(
    "path/to/document.pdf",
    strategy=SplitStrategy.FIXED_SIZE,
    chunk_size=500,
    overlap=50
)

# Index with sentence strategy (each sentence separately)
num_chunks, chunks = indexer.index_document(
    "path/to/document.docx",
    strategy=SplitStrategy.SENTENCE,
    sentences_per_chunk=1
)

# Index with paragraph strategy
num_chunks, chunks = indexer.index_document(
    "path/to/document.pdf",
    strategy=SplitStrategy.PARAGRAPH,
    paragraphs_per_chunk=1
)
```

## 📊 Splitting Strategies

### 1. Fixed-Size with Overlap
Splits text into fixed-size chunks (500 characters) with overlap (50 characters) between them.
- **Advantage:** Consistent chunk sizes
- **Best for:** Long, uniform texts
- **Note:** The module prevents creation of unnecessarily small chunks at document end

### 2. Sentence-Based Splitting
Splits text by sentences - each sentence becomes a separate chunk.
- **Advantage:** Preserves sentence integrity
- **Best for:** Documents with clear sentences
- **Note:** The module normalizes whitespace and newlines from PDFs for correct sentence detection

### 3. Paragraph-Based Splitting
Splits text by paragraphs.
- **Advantage:** Preserves semantic context
- **Best for:** Articles, reports, structured documents
- **Note:** The module identifies paragraphs even in PDF documents where each line ends with a newline, by detecting headers (like "Title:") or blank lines

## 📤 Module Output

The module prints detailed information to the console during processing:

### Steps 1-3: Text extraction, chunk splitting, and embedding generation

### Step 4: Chunks Preview
```
============================================================
CHUNKS PREVIEW
============================================================

--- Chunk 1/5 ---
Length: 487 characters
----------------------------------------
[chunk content]
----------------------------------------
```

### Step 5: Embeddings Information
```
============================================================
EMBEDDINGS PREVIEW
============================================================

--- Embedding 1/5 ---
Embedding vector length: 768
First 10 values: [0.0234, -0.0156, 0.0891, ...]
----------------------------------------
```

### Step 6: Storing in PostgreSQL

## 🗄️ Database Schema

The `document_chunks` table contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Unique identifier |
| `chunk_text` | TEXT | The chunk text |
| `embedding` | FLOAT8[] | The embedding vector |
| `filename` | VARCHAR(255) | Original filename |
| `split_strategy` | VARCHAR(50) | The splitting strategy used |
| `created_at` | TIMESTAMP | Date added |

## 📁 Project Structure

```
document-indexer/
├── index_documents.py    # Main script
├── requirements.txt      # Project dependencies
├── .env.example         # Environment variables template
├── .env                 # Environment variables (not saved in Git)
├── .gitignore          # Files to ignore
└── README.md           # This file
```

## 🏗️ Architecture

The module is built with a modular architecture with the following classes:

| Class | Purpose |
|-------|---------|
| `TextExtractor` | Extract text from PDF and DOCX |
| `TextSplitter` | Split text using different strategies |
| `EmbeddingGenerator` | Generate embeddings with Gemini API |
| `PostgresStorage` | Store and retrieve from PostgreSQL |
| `DocumentIndexer` | Orchestrate the complete process |

## 🔐 Security

- API keys and connection details are stored only in the `.env` file
- The `.env` file is listed in `.gitignore` and is not uploaded to Git
- Never share your API keys!

## 🔍 SQL Query Examples

```sql
-- Display all chunks from a specific file
SELECT id, LEFT(chunk_text, 100) as preview, split_strategy 
FROM document_chunks 
WHERE filename = 'document.pdf';

-- Count chunks by strategy
SELECT split_strategy, COUNT(*) as count 
FROM document_chunks 
GROUP BY split_strategy;

-- Delete chunks from a specific file
DELETE FROM document_chunks WHERE filename = 'document.pdf';
```

## 🛠️ Troubleshooting

### PostgreSQL Connection Error
```
Error connecting to PostgreSQL
```
**Solution:** Make sure PostgreSQL is running and the URL is correct in the `.env` file

### API Key Error
```
Gemini API key not found
```
**Solution:** Make sure you've set `GEMINI_API_KEY` in the `.env` file

### Unsupported File Format Error
```
Unsupported file format
```
**Solution:** The module only supports PDF and DOCX files

### Incorrect PDF Splitting
If text is split by lines instead of paragraphs/sentences:
- Make sure you're using the latest version of the module
- The module automatically normalizes newlines from PDFs
