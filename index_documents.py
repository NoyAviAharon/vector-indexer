"""
index_documents.py - Document Indexing Module with Vector Embeddings

This module extracts text from PDF/DOCX files, splits it into chunks using
various strategies, generates embeddings via Google Gemini API, and stores
the results in PostgreSQL.
"""

import os
import re
import argparse
from datetime import datetime
from typing import List, Tuple, Optional
from enum import Enum

# Third-party imports
import PyPDF2
from docx import Document
import google.generativeai as genai
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SplitStrategy(Enum):
    """Enumeration of available text splitting strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


class TextExtractor:
    """Handles text extraction from different file formats."""

    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text as a string
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {e}")

        return TextExtractor._clean_text(text)

    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """
        Extract text from a DOCX file.

        Args:
            file_path: Path to the DOCX file

        Returns:
            Extracted text as a string
        """
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise RuntimeError(f"Error extracting text from DOCX: {e}")

        return TextExtractor._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and special characters.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    @staticmethod
    def extract(file_path: str) -> str:
        """
        Extract text from a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text

        Raises:
            ValueError: If file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.pdf':
            return TextExtractor.extract_from_pdf(file_path)
        elif ext == '.docx':
            return TextExtractor.extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: .pdf, .docx")


class TextSplitter:
    """Handles text splitting using various strategies."""

    @staticmethod
    def split_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if overlap < 0:
            raise ValueError("Overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()

            # Skip if chunk is empty
            if not chunk:
                break

            # Skip if this chunk is too small and overlaps significantly with the previous one
            # (i.e., it's just leftover text from overlap)
            if chunks and len(chunk) < overlap:
                break

            chunks.append(chunk)

            # If we've reached the end of the text, stop
            if end >= text_length:
                break

            # Move start position, accounting for overlap
            start = end - overlap

        return chunks

    @staticmethod
    def split_by_sentences(text: str, sentences_per_chunk: int = 1) -> List[str]:
        """
        Split text by sentences.

        Args:
            text: Text to split
            sentences_per_chunk: Number of sentences per chunk (default: 1 = each sentence is a chunk)

        Returns:
            List of text chunks
        """
        # Normalize whitespace - replace newlines with spaces for better sentence detection
        normalized_text = re.sub(r'\s+', ' ', text)

        # Split by sentence-ending punctuation followed by space
        # This pattern captures the punctuation as part of the sentence
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, normalized_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # If regex didn't split well, try alternative approach
        if len(sentences) <= 1 and len(text) > 100:
            # Alternative: split on . ! ? followed by capital letter or end
            sentences = re.split(r'([.!?])\s+', normalized_text)
            # Reconstruct sentences with their punctuation
            reconstructed = []
            i = 0
            while i < len(sentences):
                if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                    reconstructed.append(sentences[i] + sentences[i + 1])
                    i += 2
                else:
                    if sentences[i].strip():
                        reconstructed.append(sentences[i].strip())
                    i += 1
            sentences = reconstructed

        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if chunk:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def split_by_paragraphs(text: str, paragraphs_per_chunk: int = 1) -> List[str]:
        """
        Split text by paragraphs.

        Args:
            text: Text to split
            paragraphs_per_chunk: Number of paragraphs per chunk

        Returns:
            List of text chunks
        """
        # First, try to detect paragraph boundaries
        # In PDFs, paragraphs are often separated by:
        # 1. Double newlines (blank lines)
        # 2. A sentence ending (.) followed by a title/header pattern (word + ":")

        # Normalize: replace single newlines (not followed by newline) with spaces
        # This handles PDFs where each line ends with \n
        normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Now split by double newlines (actual paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', normalized)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If we only got one paragraph, try alternative: split by section headers
        # Pattern: sentence ending with period, followed by a Title (capitalized word(s) + colon)
        if len(paragraphs) <= 1 and len(text) > 200:
            # Try splitting by header pattern: "Title:" or "Title Word:"
            header_pattern = r'(?<=\.)\s+(?=[A-Z][a-zA-Z\s]+:)'
            paragraphs = re.split(header_pattern, normalized)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If still only one paragraph, try splitting by sentences ending with period
        # followed by a capital letter starting a new logical section
        if len(paragraphs) <= 1 and len(text) > 200:
            # Last resort: split on double spaces or period + multiple spaces
            paragraphs = re.split(r'\.\s{2,}', normalized)
            paragraphs = [p.strip() + '.' if not p.strip().endswith('.') else p.strip()
                         for p in paragraphs if p.strip()]

        chunks = []
        for i in range(0, len(paragraphs), paragraphs_per_chunk):
            chunk = '\n\n'.join(paragraphs[i:i + paragraphs_per_chunk])
            if chunk:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def split(text: str, strategy: SplitStrategy, **kwargs) -> List[str]:
        """
        Split text using the specified strategy.

        Args:
            text: Text to split
            strategy: Splitting strategy to use
            **kwargs: Additional arguments for the splitting method

        Returns:
            List of text chunks
        """
        if strategy == SplitStrategy.FIXED_SIZE:
            return TextSplitter.split_fixed_size(text, **kwargs)
        elif strategy == SplitStrategy.SENTENCE:
            return TextSplitter.split_by_sentences(text, **kwargs)
        elif strategy == SplitStrategy.PARAGRAPH:
            return TextSplitter.split_by_paragraphs(text, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class EmbeddingGenerator:
    """Handles embedding generation using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            api_key: Google Gemini API key (optional, can be loaded from env)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
                "or pass it as an argument."
            )

        genai.configure(api_key=self.api_key)
        self.model = "models/embedding-001"

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}")

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks.

        Args:
            chunks: List of text chunks

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, chunk in enumerate(chunks):
            print(f"Generating embedding {i + 1}/{len(chunks)}...")
            embedding = self.generate_embedding(chunk)
            embeddings.append(embedding)
        return embeddings


class PostgresStorage:
    """Handles storage of chunks and embeddings in PostgreSQL."""

    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize PostgreSQL storage.

        Args:
            connection_url: PostgreSQL connection URL (optional, can be loaded from env)
        """
        self.connection_url = connection_url or os.getenv('POSTGRES_URL')

        if not self.connection_url:
            raise ValueError(
                "PostgreSQL connection URL not found. Please set POSTGRES_URL environment "
                "variable or pass it as an argument."
            )

        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(self.connection_url)
            self.cursor = self.conn.cursor()
            print("Successfully connected to PostgreSQL database.")
        except Exception as e:
            raise RuntimeError(f"Error connecting to PostgreSQL: {e}")

    def create_table(self):
        """Create the document_chunks table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding FLOAT8[] NOT NULL,
            filename VARCHAR(255) NOT NULL,
            split_strategy VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create index for faster similarity searches (optional but recommended)
        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_document_chunks_filename 
        ON document_chunks(filename);
        """

        try:
            self.cursor.execute(create_table_query)
            self.cursor.execute(create_index_query)
            self.conn.commit()
            print("Table 'document_chunks' created/verified successfully.")
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Error creating table: {e}")

    def store_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        filename: str,
        strategy: str
    ) -> int:
        """
        Store chunks and their embeddings in the database.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            filename: Original filename
            strategy: Splitting strategy used

        Returns:
            Number of rows inserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        insert_query = """
        INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy, created_at)
        VALUES %s
        """

        # Prepare data for batch insert
        data = [
            (chunk, embedding, filename, strategy, datetime.now())
            for chunk, embedding in zip(chunks, embeddings)
        ]

        try:
            execute_values(self.cursor, insert_query, data)
            self.conn.commit()
            print(f"Successfully stored {len(chunks)} chunks in the database.")
            return len(chunks)
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Error storing chunks: {e}")

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed.")


class DocumentIndexer:
    """Main class that orchestrates the document indexing process."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        postgres_url: Optional[str] = None
    ):
        """
        Initialize the document indexer.

        Args:
            gemini_api_key: Google Gemini API key
            postgres_url: PostgreSQL connection URL
        """
        self.embedding_generator = EmbeddingGenerator(gemini_api_key)
        self.storage = PostgresStorage(postgres_url)

    def index_document(
        self,
        file_path: str,
        strategy: SplitStrategy = SplitStrategy.FIXED_SIZE,
        **split_kwargs
    ) -> Tuple[int, List[str]]:
        """
        Index a document: extract text, split, generate embeddings, and store.

        Args:
            file_path: Path to the document file
            strategy: Text splitting strategy
            **split_kwargs: Additional arguments for the splitting method

        Returns:
            Tuple of (number of chunks stored, list of chunks)
        """
        # Get filename from path
        filename = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"Indexing document: {filename}")
        print(f"Strategy: {strategy.value}")
        print(f"{'='*60}\n")

        # Step 1: Extract text
        print("Step 1: Extracting text from document...")
        text = TextExtractor.extract(file_path)
        print(f"Extracted {len(text)} characters of text.\n")

        # Step 2: Split text into chunks
        print("Step 2: Splitting text into chunks...")
        chunks = TextSplitter.split(text, strategy, **split_kwargs)
        print(f"Created {len(chunks)} chunks.\n")

        if not chunks:
            print("Warning: No chunks created. Document may be empty.")
            return 0, []

        # Step 3: Generate embeddings
        print("Step 3: Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        print(f"Generated {len(embeddings)} embeddings.\n")

        # Step 4: Print chunks to console
        print("Step 4: Printing chunks to console...")
        print("\n" + "="*60)
        print("CHUNKS PREVIEW")
        print("="*60)
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i + 1}/{len(chunks)} ---")
            print(f"Length: {len(chunk)} characters")
            print("-" * 40)
            print(chunk)
            print("-" * 40)
        print("\n" + "="*60 + "\n")

        # Step 5: Print embedding vector information
        print("Step 5: Embedding vector information...")
        print("\n" + "="*60)
        print("EMBEDDINGS PREVIEW")
        print("="*60)
        for i, embedding in enumerate(embeddings):
            print(f"\n--- Embedding {i + 1}/{len(embeddings)} ---")
            print(f"Embedding vector length: {len(embedding)}")
            print(f"First 10 values: {embedding[:10]}")
            print("-" * 40)
        print("\n" + "="*60 + "\n")

        # Step 6: Store in database
        print("Step 6: Storing chunks and embeddings in PostgreSQL...")
        self.storage.connect()
        self.storage.create_table()
        num_stored = self.storage.store_chunks(
            chunks, embeddings, filename, strategy.value
        )
        self.storage.close()

        print(f"\n{'='*60}")
        print(f"Indexing complete! Stored {num_stored} chunks.")
        print(f"{'='*60}\n")

        return num_stored, chunks


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Index documents by creating vector embeddings and storing in PostgreSQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a PDF with fixed-size chunks (default)
  python index_documents.py document.pdf

  # Index a DOCX with sentence-based splitting
  python index_documents.py document.docx --strategy sentence

  # Index with paragraph-based splitting
  python index_documents.py document.pdf --strategy paragraph

  # Custom fixed-size parameters
  python index_documents.py document.pdf --strategy fixed_size --chunk-size 1000 --overlap 100
        """
    )

    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the document file (PDF or DOCX)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['fixed_size', 'sentence', 'paragraph'],
        default='fixed_size',
        help='Text splitting strategy (default: fixed_size)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Chunk size for fixed_size strategy (default: 500)'
    )

    parser.add_argument(
        '--overlap',
        type=int,
        default=50,
        help='Overlap size for fixed_size strategy (default: 50)'
    )

    parser.add_argument(
        '--sentences-per-chunk',
        type=int,
        default=1,
        help='Sentences per chunk for sentence strategy (default: 1 = each sentence is a separate chunk)'
    )

    parser.add_argument(
        '--paragraphs-per-chunk',
        type=int,
        default=1,
        help='Paragraphs per chunk for paragraph strategy (default: 1)'
    )

    args = parser.parse_args()

    # Map string to enum
    strategy_map = {
        'fixed_size': SplitStrategy.FIXED_SIZE,
        'sentence': SplitStrategy.SENTENCE,
        'paragraph': SplitStrategy.PARAGRAPH
    }
    strategy = strategy_map[args.strategy]

    # Prepare kwargs based on strategy
    split_kwargs = {}
    if strategy == SplitStrategy.FIXED_SIZE:
        split_kwargs = {
            'chunk_size': args.chunk_size,
            'overlap': args.overlap
        }
    elif strategy == SplitStrategy.SENTENCE:
        split_kwargs = {'sentences_per_chunk': args.sentences_per_chunk}
    elif strategy == SplitStrategy.PARAGRAPH:
        split_kwargs = {'paragraphs_per_chunk': args.paragraphs_per_chunk}

    # Run the indexer
    try:
        indexer = DocumentIndexer()
        num_chunks, chunks = indexer.index_document(
            args.file_path,
            strategy=strategy,
            **split_kwargs
        )

        # Print sample of first chunk
        if chunks:
            print("\nSample of first chunk:")
            print("-" * 40)
            print(chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0])
            print("-" * 40)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        exit(1)
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()