# Project 1 — HR Policy Assistant (RAG System)

A local Retrieval-Augmented Generation (RAG) system for querying employee human resource policies, built using LangChain, ChromaDB, HuggingFace embeddings (`all-MiniLM-L6-v2`), and an interactive Gradio Web UI.

---

## 🏗️ Architecture & Pipeline Flow

```
   ┌──────────────────┐
   │  hr_manual.pdf   │
   └────────┬─────────┘
            │
            ▼ (PyPDFLoader)
   ┌──────────────────┐
   │ Loaded Documents │
   └────────┬─────────┘
            │
            ▼ (RecursiveCharacterTextSplitter: chunk_size=500, overlap=50)
   ┌──────────────────┐
   │ Text Chunks (11) │
   └────────┬─────────┘
            │
            ▼ (HuggingFaceEmbeddings: all-MiniLM-L6-v2)
   ┌───────────────────────────┐
   │ ChromaDB Vector Database  │ <─── Persisted in ./chroma_db
   └─────────────┬─────────────┘
                 │
                 ▼ (Similarity Search with Distance Threshold)
   ┌───────────────────────────────────────────────┐
   │  Grounded QA Engine & Strict Filter (app.py)   │
   │  - Strict Grounding with Page/Section Citations│
   │  - Rejection with "Not found" if out-of-domain│
   └─────────────┬─────────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────┐   ┌────────────────────────┐
│  CLI Runner  │   │ Interactive Gradio UI  │
│  (app.py)    │   │ (ui.py -> :7860)       │
└──────────────┘   └────────────────────────┘
```

1. **PDF Ingestion:** `PyPDFLoader` parses the HR manual and extracts page-level metadata.
2. **Text Chunking:** `RecursiveCharacterTextSplitter` segments text into 500-character chunks with 50-character overlap across semantic boundaries.
3. **Embeddings & Persistence:** Embeddings are generated using HuggingFace's `all-MiniLM-L6-v2` (384-dimensional vectors) and stored locally in `./chroma_db`.
4. **Retrieval & Strict Grounding:** Queries vector store with distance metrics. If an inquiry is out of scope or absent (e.g., GST rates), it strictly outputs `"Not found"`.
5. **Exact References:** Every valid answer provides exact page numbers and section citations.

---

## 📋 Evaluation Questions & Answers

| # | Question | Grounded Answer | Reference Citation |
|---|---|---|---|
| **1** | How many days of Privilege Leave (PL) does a permanent employee get per year? | Permanent employees are entitled to eighteen (18) days of Privilege Leave (PL) per calendar year, accrued at the rate of 1.5 days per completed month of active service. | **Page 1, Section 3.1: Privilege Leave (PL)** |
| **2** | What is the probation period for employees up to Senior Manager? | The probation period for all new employees up to the grade of Senior Manager is six (6) months from the official date of joining. | **Page 1, Section 2.2: Duration of Probation** |
| **3** | What is the company's health-insurance premium amount? | The company pays an annual health-insurance premium amount of INR 15,000 per employee family unit directly to the insurer. Employees have zero co-contribution towards this base premium. | **Page 2, Section 4.1: Group Medical Health Insurance** |
| **4** | What is the current GST rate in India? | **Not found** | **None** *(Topic not covered in HR Manual)* |
| **5** | How many days of maternity leave are given? | Eligible female employees are entitled to twenty-six (26) weeks, equivalent to one hundred eighty-two (182) days, of fully paid maternity leave for up to the first two surviving children. | **Page 2, Section 3.3: Maternity and Parental Leave** |

---

## 🚀 Getting Started

### 1. Installation
Install the necessary dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Evaluation via CLI
Run the automated evaluation suite against the 5 questions:
```bash
python3 app.py
```

To run a custom question directly in CLI:
```bash
python3 app.py "What is the notice period during probation?"
```

### 3. Launch the Interactive Gradio Web UI
Start the local web application:
```bash
python3 ui.py
```
Open your browser and navigate to: **`http://127.0.0.1:7860`**

Features of the Web UI:
- **🔍 Policy Q&A Tab:** Enter any question, click one of the 5 evaluation examples, view real-time answer cards, exact reference badges, and inspect underlying vector context chunks.
- **💬 Conversational Assistant Tab:** Chat conversationally with the HR Policy Assistant with multi-turn memory.
- **📖 Document Overview Tab:** Full summary of policy sections covered in `hr_manual.pdf`.
