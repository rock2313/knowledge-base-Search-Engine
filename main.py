from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from datetime import datetime
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
from pathlib import Path
import json
import re  # Added for regex operations

# Initialize FastAPI
app = FastAPI(title="Knowledge Base Search Engine")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_PAGES = 1000
CHUNK_SIZE = 100  # Smaller chunks for better granularity
CHUNK_OVERLAP = 20  # Smaller overlap

# Initialize embedding model (free, runs locally)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create collection
try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

# Document metadata storage
documents_db = {}

# OpenAI API key (set via environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# Models
class SearchQuery(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5

class SearchResponse(BaseModel):
    answer: str
    sources: List[dict]
    processing_time: float

# Helper functions
def extract_text_from_pdf(file_path: str, max_pages: int = MAX_PAGES) -> tuple:
    """Extract text from PDF file"""
    text_chunks = []
    page_texts = []

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = min(len(pdf_reader.pages), max_pages)

            if len(pdf_reader.pages) > max_pages:
                print(f"Warning: PDF has {len(pdf_reader.pages)} pages, limiting to {max_pages}")

            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                page_texts.append({"page": page_num + 1, "text": text})

                # Chunk the text
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                for chunk_idx, chunk in enumerate(chunks):
                    text_chunks.append({
                        "text": chunk,
                        "page": page_num + 1,
                        "chunk_id": f"{page_num + 1}_{chunk_idx}"
                    })
    except Exception as e:
        raise Exception(f"Error extracting PDF: {str(e)}")

    return text_chunks, total_pages

def extract_text_from_txt(file_path: str) -> tuple:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        text_chunks = [
            {"text": chunk, "page": 1, "chunk_id": f"1_{idx}"}
            for idx, chunk in enumerate(chunks)
        ]
        return text_chunks, 1
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, prioritizing natural boundaries"""

    chunks = []

    # First, try to split by clear section markers (SECTION, ###, ---, etc.)
    section_pattern = r'(SECTION \d+:|#{2,}|---+|\n\n)'
    sections = re.split(section_pattern, text)

    current_chunk = []
    current_word_count = 0

    for section in sections:
        section = section.strip()
        if not section or section in ['SECTION', '##', '###', '---']:
            continue

        words = section.split()
        word_count = len(words)

        # If this section alone is bigger than chunk_size, split it further
        if word_count > chunk_size:
            # Save current chunk if exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split large section into smaller chunks
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) > 10:  # Minimum chunk size
                    chunks.append(' '.join(chunk_words))
        else:
            # Check if adding this section would exceed chunk_size
            if current_word_count + word_count > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                    current_word_count = overlap
                else:
                    current_chunk = []
                    current_word_count = 0

            # Add section to current chunk
            current_chunk.extend(words)
            current_word_count += word_count

    # Add final chunk
    if current_chunk and len(current_chunk) > 10:
        chunks.append(' '.join(current_chunk))

    # Fallback: if no chunks created, do simple word-based splitting
    if not chunks:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 10:
                chunks.append(' '.join(chunk_words))

    return chunks if chunks else [text]

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using sentence transformers"""
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()

def synthesize_answer(query: str, contexts: List[str], mode: str = "search") -> str:
    """Use LLM to synthesize answer from retrieved contexts"""

    if not openai.api_key or openai.api_key == "":
        # Improved fallback: create a basic synthesis without LLM
        # Use only the most relevant context (first one)
        best_context = contexts[0] if contexts else ""

        if mode == "summarize":
            return f"📄 Based on the retrieved information about '{query}':\n\n{best_context[:600]}\n\n💡 Note: Set OPENAI_API_KEY for AI-powered synthesis."
        else:
            # Clean and format the response
            clean_context = best_context.strip()
            # Remove excessive whitespace
            clean_context = ' '.join(clean_context.split())

            return f"Based on the query '{query}':\n\n{clean_context[:500]}\n\n💡 Note: Set OPENAI_API_KEY for AI-powered answer synthesis."

    context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

    if mode == "summarize":
        prompt = f"""Based on the following document excerpts, provide a comprehensive summary about: {query}

Document Excerpts:
{context_text}

Provide a well-structured summary that covers the key points, main ideas, and important details related to the topic."""
    else:
        prompt = f"""Using the following document excerpts, answer the user's question accurately and succinctly.

Question: {query}

Document Excerpts:
{context_text}

Answer: Provide a clear, accurate answer based on the excerpts above. If the excerpts don't contain enough information to answer the question, acknowledge the limitations."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that synthesizes information from documents to answer questions accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback if API call fails
        best_context = contexts[0] if contexts else "No context available"
        clean_context = ' '.join(best_context.split())
        return f"⚠️ LLM synthesis unavailable.\n\nMost relevant excerpt:\n\n{clean_context[:400]}..."

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Knowledge Base Search Engine API",
        "version": "1.0.0",
        "endpoints": ["/upload", "/search", "/summarize", "/documents"]
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""

    # Validate file type
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(400, "Only PDF and TXT files are supported")

    # Generate unique document ID
    doc_id = str(uuid.uuid4())

    # Save file
    file_ext = file.filename.split('.')[-1]
    file_path = UPLOAD_DIR / f"{doc_id}.{file_ext}"

    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    try:
        # Extract text
        if file_ext == 'pdf':
            text_chunks, total_pages = extract_text_from_pdf(str(file_path), MAX_PAGES)
        else:
            text_chunks, total_pages = extract_text_from_txt(str(file_path))

        if not text_chunks:
            raise HTTPException(400, "No text could be extracted from document")

        # Generate embeddings
        texts = [chunk["text"] for chunk in text_chunks]
        embeddings = generate_embeddings(texts)

        # Store in ChromaDB
        ids = [f"{doc_id}_{chunk['chunk_id']}" for chunk in text_chunks]
        metadatas = [
            {
                "document_id": doc_id,
                "document_name": file.filename,
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in text_chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        # Store document metadata
        documents_db[doc_id] = {
            "id": doc_id,
            "name": file.filename,
            "pages": total_pages,
            "chunks": len(text_chunks),
            "uploaded_at": datetime.now().isoformat(),
            "file_path": str(file_path)
        }

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "pages": total_pages,
            "chunks": len(text_chunks),
            "message": "Document uploaded and indexed successfully"
        }

    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Error processing document: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Search across documents and synthesize answer"""

    import time
    start_time = time.time()

    # Validate query
    if not query.query or query.query.strip() == "":
        raise HTTPException(400, "Query cannot be empty")

    # Check if there are any documents
    if collection.count() == 0:
        raise HTTPException(404, "No documents available. Please upload documents first.")

    # Generate query embedding
    print(f"🔍 Searching for: '{query.query}'")
    query_embedding = embedding_model.encode([query.query])[0].tolist()

    # Build filter
    where_filter = None
    if query.document_ids:
        where_filter = {"document_id": {"$in": query.document_ids}}
        print(f"📁 Filtering by document IDs: {query.document_ids}")

    # Query ChromaDB with semantic search - get more results to filter
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(query.top_k * 2, collection.count()),  # Get extra results to filter
        where=where_filter
    )

    if not results['documents'][0]:
        raise HTTPException(404, "No relevant documents found for this query")

    print(f"✅ Found {len(results['documents'][0])} relevant chunks")

    # Prepare sources with relevance filtering
    sources = []
    contexts = []

    # Calculate similarity threshold - only keep highly relevant chunks
    min_similarity = 0.3  # Minimum 30% similarity

    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        similarity = max(0, 1 - distance)

        # Only include chunks above similarity threshold
        if similarity >= min_similarity:
            contexts.append(text)
            sources.append({
                "text": text[:300] + "..." if len(text) > 300 else text,
                "document_name": metadata['document_name'],
                "page": metadata['page'],
                "score": similarity
            })

    if not sources:
        raise HTTPException(404, "No sufficiently relevant content found")

    # Sort sources by relevance score
    sources.sort(key=lambda x: x['score'], reverse=True)

    # Limit to top results for answer synthesis
    top_contexts = contexts[:3]  # Only use top 3 most relevant chunks
    top_sources = sources[:query.top_k]  # Return requested number of sources

    print(f"📊 Using top {len(top_contexts)} contexts for synthesis")

    # Synthesize answer using top contexts only
    answer = synthesize_answer(query.query, top_contexts, mode="search")

    processing_time = time.time() - start_time
    print(f"⏱️ Processing time: {processing_time:.2f}s")

    return SearchResponse(
        answer=answer,
        sources=top_sources,
        processing_time=processing_time
    )

@app.post("/summarize", response_model=SearchResponse)
async def summarize(query: SearchQuery):
    """Generate summary on a specific topic"""

    import time
    start_time = time.time()

    # Generate query embedding
    query_embedding = embedding_model.encode([query.query])[0].tolist()

    # Build filter
    where_filter = None
    if query.document_ids:
        where_filter = {"document_id": {"$in": query.document_ids}}

    # Query ChromaDB with higher top_k for summaries
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(query.top_k * 2, 15),
        where=where_filter
    )

    if not results['documents'][0]:
        raise HTTPException(404, "No relevant documents found")

    # Prepare sources
    sources = []
    contexts = []

    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]

        contexts.append(text)
        sources.append({
            "text": text[:300] + "..." if len(text) > 300 else text,
            "document_name": metadata['document_name'],
            "page": metadata['page'],
            "score": 1 - distance
        })

    # Synthesize summary
    answer = synthesize_answer(query.query, contexts, mode="summarize")

    processing_time = time.time() - start_time

    return SearchResponse(
        answer=answer,
        sources=sources,
        processing_time=processing_time
    )

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {"documents": list(documents_db.values())}

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""

    if document_id not in documents_db:
        raise HTTPException(404, "Document not found")

    # Delete from ChromaDB
    # Get all chunk IDs for this document
    results = collection.get(
        where={"document_id": document_id}
    )

    if results['ids']:
        collection.delete(ids=results['ids'])

    # Delete file
    file_path = Path(documents_db[document_id]['file_path'])
    if file_path.exists():
        file_path.unlink()

    # Remove from metadata
    del documents_db[document_id]

    return {"message": "Document deleted successfully"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "documents_count": len(documents_db),
        "embeddings_count": collection.count()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)