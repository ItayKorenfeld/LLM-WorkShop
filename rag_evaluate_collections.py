import os
import json
from typing import List, Dict

import pandas as pd
from openai import OpenAI

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------

PERSIST_DIR = "./chroma_db_test"

COLLECTIONS = {
    "lectures": "ds_lectures",
    "exams": "ds_exams",
    "combined": "ds_combined"
}

TEST_QUESTIONS_JSON = "./Test_Questions_TLV.json"
OUTPUT_EXCEL = "4_rag_evaluation_results.xlsx"

TOP_K = 2
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5"
TEMPERATURE = 0
PLAIN_COL = "plain_response"
PLAIN_COL_QUESTION = "plain_question"
GRADER_MODEL = "gpt-4o"

os.environ["CHROMA_TELEMETRY"] = "False"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# ---------------- OPENAI ----------------

llm_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- CHROMA SETUP ----------------

chroma_client = PersistentClient(
    settings=Settings(
        persist_directory=PERSIST_DIR
    )
)

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL
)

collections = {
    name: chroma_client.get_or_create_collection(
        name=coll_name,
        embedding_function=embedding_fn
    )
    for name, coll_name in COLLECTIONS.items()
}

# ---------------- RAG HELPERS ----------------

def retrieve_context(collection, question: str, k: int) -> List[str]:
    """
    Retrieve top-k documents from a Chroma collection.
    """
    results = collection.query(
        query_texts=[question],
        n_results=k
    )

    return results["documents"][0]


def build_prompt(question: str, contexts: List[str]) -> List[Dict]:
    """
    Builds a RAG prompt for GPT-5.
    """
    
    if not contexts:
        return [
        {
            "role": "system",
            "content": "You are a helpful assistant solving Data Structures exam questions."
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Answer the question clearly and concisely.
""".strip()
        }
    ]
    context_text = "\n\n".join(
        f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)
    )

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant solving Data Structures exam questions."
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Relevant material:
{context_text}

Answer the question clearly and concisely.
""".strip()
        }
    ]


def call_llm(messages: List[Dict]) -> str:
    """
    Calls GPT-5 safely.
    """
    response = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
       
    )
    return response.choices[0].message.content.strip()
def grade_answer(
    official_answer: str,
    model_answer: str
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a professor that acts as an exam grader."
        },
        {
            "role": "user",
            "content": f"""

Official Answer:
{official_answer}

Student Answer:
{model_answer}

Grade the student answer from 0 to 100 (any integer value in that range is legal) with 60 percent on the final answer and 40 percent on the explanation and the way.
Output ONLY the number.
""".strip()
        }
    ]

    response = llm_client.chat.completions.create(
        model=GRADER_MODEL,
        messages=messages
    )

    text = response.choices[0].message.content.strip()

    try:
        return text
    except ValueError:
        return None

# ---------------- EVALUATION LOOP ----------------

with open(TEST_QUESTIONS_JSON, "r", encoding="utf-8") as f:
    test_questions = json.load(f)

rows = []
grade_rows = []


for idx, item in enumerate(test_questions, 1):
    qid = item["id"]
    qtext = item["original_question_text"]
    qanswer = item["official_answer"]

    print(f"Processing {qid}")
    row = {
        "QuestionID": qid
    }
    grade_row = {
        "QuestionID": qid
    }

    for coll_key, collection in collections.items():
        contexts = retrieve_context(collection, qtext, TOP_K)
        prompt = build_prompt(qtext, contexts)
        answer = call_llm(prompt)
        grade=grade_answer(qanswer, answer)
        grade_row[f"{coll_key}_grade"] = grade

        row[f"{coll_key}_response"] = answer
        row[f"{coll_key}_contexts"] = prompt[1]["content"]
    prompt=build_prompt(qtext, None)
    answer = call_llm(prompt)
    grade=grade_answer(qanswer, answer)
    grade_row[PLAIN_COL+"_grade"] = grade
    row[PLAIN_COL] = answer
    row[PLAIN_COL_QUESTION] = qtext

    rows.append(row)
    grade_rows.append(grade_row)
# ---------------- SAVE RESULTS ----------------

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_EXCEL, index=False)
grades_df= pd.DataFrame(grade_rows)
grades_output_excel="rag_evaluation_grades.xlsx"
grades_df.to_excel(grades_output_excel, index=False)
print(f"\nEvaluation complete. Results saved to {OUTPUT_EXCEL}")
