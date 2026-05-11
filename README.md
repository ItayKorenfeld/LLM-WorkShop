# RAG Project Setup Guide

This document describes how to set up and run the RAG (Retrieval-Augmented Generation) pipelines for the Lectures and Exams datasets, as well as how to execute evaluation scripts.

---

## 1. Directory Structure

Before running any scripts, ensure the following folder structure is created:

```
.
├── Lectures_DS/
│   └── Lectures.pdf
│
├── Exams/
│   └── exams.json
│
├── Test_Questions_TLV.json
├── <various_scripts>.py
```

---

## 2. Dataset Setup

### Lectures RAG

* Create a folder named:

  ```
  Lectures_DS/
  ```
* Place the lecture material inside it as a PDF file:

  ```
  Lectures_DS/Lectures.pdf
  ```

This PDF is used as the knowledge base for the Lectures RAG pipeline.

---

### Exams RAG

* Create a folder named:

  ```
  Exams/
  ```
* Place the dataset file inside it:

  ```
  Exams/exams.json
  ```

This JSON file is used as the knowledge base for the Exams RAG pipeline.

---

## 3. Evaluation Questions

To run evaluation across all RAG systems, ensure the following file exists in the root directory:

```
Test_Questions_TLV.json
```

This file contains the test queries used for benchmarking and comparison.

---

## 4. Running the Scripts

Each component of the system is executed using Python. To run any script, use:

```bash
python3 "{name_of_the_script}.py"
```

### Example:

```bash
python3 build_lectures_rag.py
python3 build_exams_rag.py
python3 run_evaluation.py
```

---

## 5. Notes

* Ensure all dependencies are installed before running the scripts.
* Maintain the exact folder structure, as paths are hardcoded in the pipeline.
* Any missing file will result in runtime errors during indexing or evaluation.

---

## 6. Summary

* Lectures RAG uses: `Lectures_DS/Lectures.pdf`
* Exams RAG uses: `Exams/exams.json`
* Evaluation uses: `Test_Questions_TLV.json`
* Execution: `python3 script_name.py`
