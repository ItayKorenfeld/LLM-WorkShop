import os
import json
from pathlib import Path

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------

EXAMS_JSON = "./Exams/exams.json"
PERSIST_DIR = "./chroma_db_test"
COLLECTION_NAME = "ds_exams"

EMBEDDING_MODEL = "text-embedding-3-small"

os.environ["CHROMA_TELEMETRY"] = "False"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# ---------------- CHROMA SETUP ----------------
print("Setting up ChromaDB client...")
client = PersistentClient(
    settings=Settings(
        persist_directory=PERSIST_DIR
    )
)
print("ChromaDB client setup complete.")

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL
)
print("Embedding function setup complete.")

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)
print(f"Using collection: {COLLECTION_NAME}")

# ---------------- INGEST EXAMS ----------------
with open(EXAMS_JSON, "r", encoding="utf-8") as f:
    exams = json.load(f)

count = 0

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

    print(f"Adding document ID: {doc_id}")
    collection.add(
        documents=[text],
        metadatas=[{
            "source": "exam",
            "exam": exam,
            "question_number": qnum
        }],
        ids=[doc_id]
    )

    count += 1

print(f"Exam ingestion complete. Questions added: {count}")
print("Collection size:", collection.count())

