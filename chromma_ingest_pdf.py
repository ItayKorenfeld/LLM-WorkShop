import os
from pathlib import Path
from typing import List
from pypdf import PdfReader
import tiktoken

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------

LECTURES_DIR = "./Lectures_DS"
PERSIST_DIR = "./chroma_db_test"
COLLECTION_NAME = "ds_lectures"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 80

os.environ["CHROMA_TELEMETRY"] = "False"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# ---------------- UTILS ----------------

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def chunk_text(text: str) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + CHUNK_TOKENS
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start = end - CHUNK_OVERLAP

    return chunks

def extract_pdf_text(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return pages_text

# ---------------- CHROMA SETUP ----------------

client = PersistentClient(
    settings=Settings(
        persist_directory=PERSIST_DIR
    )
)

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# ---------------- INGESTION ----------------

doc_counter = 0

for pdf_file in Path(LECTURES_DIR).glob("*.pdf"):
    print(f"\nProcessing: {pdf_file.name}")
    pages = extract_pdf_text(pdf_file)

    for page_num, page_text in enumerate(pages):
        chunks = chunk_text(page_text)

        for chunk_id, chunk in enumerate(chunks):
            doc_id = f"{pdf_file.stem}_p{page_num}_c{chunk_id}"

            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": "lecture",
                    "lecture": pdf_file.name,
                    "page": page_num,
                    "chunk_id": chunk_id
                }],
                ids=[doc_id]
            )

            doc_counter += 1

print(f"\nIngestion complete. Total chunks added: {doc_counter}")
print("Collection size:", collection.count())
