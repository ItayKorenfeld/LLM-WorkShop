import os
import json
from pathlib import Path
from pypdf import PdfReader
import tiktoken

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------

LECTURES_DIR = "./Lectures_DS"
EXAMS_JSON = "./Exams/exams.json"
PERSIST_DIR = "./chroma_db_test"
COLLECTION_NAME = "ds_combined"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 80

os.environ["CHROMA_TELEMETRY"] = "False"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# ---------------- TOKENIZER ----------------

tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text):
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + CHUNK_TOKENS
        chunks.append(tokenizer.decode(tokens[start:end]))
        start = end - CHUNK_OVERLAP

    return chunks

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

# ---------------- INGEST LECTURES ----------------

for pdf_file in Path(LECTURES_DIR).glob("*.pdf"):
    reader = PdfReader(str(pdf_file))

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):
            doc_id = f"lecture_{pdf_file.stem}_p{page_num}_c{chunk_id}"

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

# ---------------- INGEST EXAMS ----------------

with open(EXAMS_JSON, "r", encoding="utf-8") as f:
    exams = json.load(f)

for item in exams:
    exam = item["exam_id"]
    qnum = item["question_number"]
    question = item["question_description"]
    answer = item["answer"]

    text = f"""
Question:
{question}

Answer:
{answer}
""".strip()

    doc_id = f"{exam}_q{qnum}"

    collection.add(
        documents=[text],
        metadatas=[{
            "source": "exam",
            "exam": exam,
            "question_number": qnum
        }],
        ids=[doc_id]
    )

print("Combined ingestion complete.")
print("Collection size:", collection.count())
