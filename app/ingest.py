
import os
import sys
import time

import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

from utils import chunk_document

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not set in .env")
    sys.exit(1)
_client_ai = genai.Client(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

# Gemini's embed_content supports batching via a list of contents, but
# free-tier rate limits are tight, so we batch in small groups and pace
# requests rather than firing everything at once.
BATCH_SIZE = 10


def embed_batch(texts):
    embeddings = []
    for text in texts:
        for attempt in range(4):
            try:
                result = _client_ai.models.embed_content(
                    model=EMBED_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768
                    ),
                )
                embeddings.append(result.embeddings[0].values)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  embed retry in {wait}s ({e})")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding failed after retries")
    return embeddings


def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"No docs/ directory found at {DOCS_DIR}")
        sys.exit(1)

    pdf_files = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCS_DIR}. Add at least 5 and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDFs: {pdf_files}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # rebuild fresh each run
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0
    for fname in pdf_files:
        path = os.path.join(DOCS_DIR, fname)
        print(f"\nProcessing {fname} ...")
        chunks = chunk_document(path, source_file=fname)
        print(f"  -> {len(chunks)} chunks")

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c.text for c in batch]
            embeddings = embed_batch(texts)

            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index,
                    }
                    for c in batch
                ],
            )
            print(f"  embedded+stored chunks {i}-{i+len(batch)-1}")

        total_chunks += len(chunks)

    print(f"\nDone. {total_chunks} chunks stored in Chroma at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
