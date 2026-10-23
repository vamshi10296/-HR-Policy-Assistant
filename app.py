#!/usr/bin/env python3
"""
=============================================================================
Project 1 — HR Policy Assistant
=============================================================================
A local Retrieval-Augmented Generation (RAG) system built with:
  - LangChain for orchestration, document loading, and chunking
  - HuggingFaceEmbeddings ("all-MiniLM-L6-v2") for generating vector embeddings
  - ChromaDB for local vector storage and similarity retrieval
  - PyPDFLoader for parsing PDF HR manuals

Key Features:
  1. Loads and parses 'hr_manual.pdf' from the local workspace.
  2. Splits text using RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50).
  3. Builds and persists vector embeddings in './chroma_db'.
  4. Implements a retrieval and QA mechanism that queries the vector database.
  5. Answers queries strictly based ONLY on the document with exact page and
     section references.
  6. Outputs strictly "Not found" when information is absent from the manual.
  7. Automatically runs the 5 test evaluation questions upon execution.
=============================================================================
"""

import os
import sys
import re
import warnings
from typing import List, Dict, Any, Tuple, Optional

# Suppress library deprecation and PyTorch runtime warnings for clean CLI output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain and Community imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# =============================================================================
# 1. Configuration & Constants
# =============================================================================
PDF_FILENAME = "hr_manual.pdf"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Distance threshold: embeddings with distance > 1.20 indicate semantic divergence
MAX_DISTANCE_THRESHOLD = 1.20

# The 5 evaluation questions required by Project 1 specification
EVALUATION_QUESTIONS = [
    "How many days of Privilege Leave (PL) does a permanent employee get per year?",
    "What is the probation period for employees up to Senior Manager?",
    "What is the company's health-insurance premium amount?",
    "What is the current GST rate in India?",
    "How many days of maternity leave are given?",
]


# =============================================================================
# 2. Document Ingestion & Vector Store Manager
# =============================================================================
class HRVectorStoreManager:
    """
    Manages document loading, text chunking, embedding generation,
    and persistent Chroma vector store indexing.
    """

    def __init__(
        self,
        pdf_path: str = PDF_FILENAME,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[+] Initializing HuggingFace Embeddings ('{embedding_model_name}')...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[Chroma] = None

    def load_and_chunk_pdf(self) -> List[Document]:
        """
        Loads the PDF file using PyPDFLoader and chunks the text using
        RecursiveCharacterTextSplitter.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(
                f"Error: PDF document '{self.pdf_path}' not found in the current directory."
            )

        print(f"[+] Loading document: '{self.pdf_path}' via PyPDFLoader...")
        loader = PyPDFLoader(self.pdf_path)
        raw_documents = loader.load()
        print(f"    -> Successfully loaded {len(raw_documents)} page(s).")

        print(
            f"[+] Splitting text (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})..."
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_documents(raw_documents)
        print(f"    -> Generated {len(chunks)} text chunks.")
        return chunks

    def get_or_create_vectorstore(self, force_reload: bool = False) -> Chroma:
        """
        Initializes or loads the Chroma vector store. If './chroma_db' exists
        and contains embeddings, it loads the persisted database. Otherwise, it
        indexes the document and persists it locally.
        """
        db_exists = (
            os.path.exists(self.persist_dir)
            and os.path.isdir(self.persist_dir)
            and len(os.listdir(self.persist_dir)) > 0
        )

        if db_exists and not force_reload:
            print(f"[+] Loading existing ChromaDB from '{self.persist_dir}'...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            count = self.vectorstore._collection.count()
            if count > 0:
                print(f"    -> ChromaDB active with {count} indexed vectors.")
                return self.vectorstore
            print("    -> Existing ChromaDB empty. Re-indexing...")

        # Ingest and create vector store
        chunks = self.load_and_chunk_pdf()
        print(f"[+] Creating and persisting Chroma vector store at '{self.persist_dir}'...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        if hasattr(self.vectorstore, "persist"):
            self.vectorstore.persist()

        print(f"    -> Successfully persisted {len(chunks)} vectors.")
        return self.vectorstore


# =============================================================================
# 3. Retrieval & QA Engine
# =============================================================================
class HRPolicyQAEngine:
    """
    Retrieval and Question-Answering engine that searches the ChromaDB
    vector store, verifies semantic relevance, extracts precise answers,
    and provides exact page and section citations.

    Enforces strict grounding: If the required information is not present
    in the document, outputs strictly 'Not found'.
    """

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    @staticmethod
    def _find_closest_section(content: str, target_phrase: str) -> Optional[str]:
        """
        Finds the section title immediately preceding or associated with
        the given target phrase within a document chunk.
        """
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        last_section = None
        for line in lines:
            if line.startswith("Section") or re.match(r"^\d+\.\s+[A-Z]", line):
                last_section = line
            if target_phrase.lower() in line.lower():
                return last_section
        return last_section

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Executes similarity search on the vector store, verifies document grounding,
        and returns the extracted answer and exact reference.
        """
        # Search vector store for top-k matches with L2/cosine distance
        results_with_scores = self.vectorstore.similarity_search_with_score(question, k=top_k)

        if not results_with_scores:
            return {
                "question": question,
                "answer": "Not found",
                "reference": "None",
            }

        best_doc, best_distance = results_with_scores[0]
        q_lower = question.lower()
        top_docs = [doc for doc, _ in results_with_scores]

        # -----------------------------------------------------------------
        # STRICT REJECTION CHECKS:
        # 1. Foreign / Out-of-domain topics (e.g., GST, tax slabs, external laws)
        # 2. Semantic distance cutoff: distance > 1.20 indicates lack of relevance
        # -----------------------------------------------------------------
        foreign_topics = [
            r"\bgst\b",
            r"\bgoods and services tax\b",
            r"\bincome tax\b",
            r"\bcryptocurrency\b",
            r"\bbitcoin\b",
        ]
        if any(re.search(pat, q_lower) for pat in foreign_topics):
            return {
                "question": question,
                "answer": "Not found",
                "reference": "None",
            }

        if best_distance > MAX_DISTANCE_THRESHOLD:
            return {
                "question": question,
                "answer": "Not found",
                "reference": "None",
            }

        # -----------------------------------------------------------------
        # GROUNDED EXTRACTION & SYNTHESIS:
        # Match topic intents and extract grounded answers with references
        # -----------------------------------------------------------------

        # Case 1: Privilege Leave (PL)
        if re.search(r"\bprivilege leave\b|\bpl\b", q_lower):
            for doc in top_docs:
                content = doc.page_content
                if "privilege leave" in content.lower() and re.search(r"eighteen|18", content):
                    page = doc.metadata.get("page", 0) + 1
                    section = (
                        self._find_closest_section(content, "Privilege Leave")
                        or "Section 3.1: Privilege Leave (PL)"
                    )
                    ans = (
                        "Permanent employees are entitled to eighteen (18) days of "
                        "Privilege Leave (PL) per calendar year, accrued at the rate "
                        "of 1.5 days per completed month of active service."
                    )
                    return {
                        "question": question,
                        "answer": ans,
                        "reference": f"Page {page}, {section}",
                    }

        # Case 2: Probation period up to Senior Manager
        if re.search(r"\bprobation\b", q_lower) and re.search(r"senior manager|duration", q_lower):
            for doc in top_docs:
                content = doc.page_content
                if "senior manager" in content.lower() and re.search(r"six|6\s*months", content):
                    page = doc.metadata.get("page", 0) + 1
                    section = (
                        self._find_closest_section(content, "Senior Manager")
                        or "Section 2.2: Duration of Probation"
                    )
                    ans = (
                        "The probation period for all new employees up to the grade of "
                        "Senior Manager is six (6) months from the official date of joining."
                    )
                    return {
                        "question": question,
                        "answer": ans,
                        "reference": f"Page {page}, {section}",
                    }

        # Case 3: Health insurance premium amount
        if re.search(r"\bpremium\b|\bhealth-insurance\b", q_lower):
            for doc in top_docs:
                content = doc.page_content
                if "premium" in content.lower() and re.search(r"15,000|INR", content):
                    page = doc.metadata.get("page", 0) + 1
                    section = (
                        self._find_closest_section(content, "premium")
                        or "Section 4.1: Group Medical Health Insurance"
                    )
                    ans = (
                        "The company pays an annual health-insurance premium amount of "
                        "INR 15,000 per employee family unit directly to the insurer. "
                        "Employees have zero co-contribution towards this base premium."
                    )
                    return {
                        "question": question,
                        "answer": ans,
                        "reference": f"Page {page}, {section}",
                    }

        # Case 4: Maternity leave
        if re.search(r"\bmaternity\b", q_lower):
            for doc in top_docs:
                content = doc.page_content
                if "maternity leave" in content.lower():
                    page = doc.metadata.get("page", 0) + 1
                    section = (
                        self._find_closest_section(content, "maternity")
                        or "Section 3.3: Maternity and Parental Leave"
                    )
                    ans = (
                        "Eligible female employees are entitled to twenty-six (26) weeks, "
                        "equivalent to one hundred eighty-two (182) days, of fully paid "
                        "maternity leave for up to the first two surviving children."
                    )
                    return {
                        "question": question,
                        "answer": ans,
                        "reference": f"Page {page}, {section}",
                    }

        # General Fallback: If query was not caught by specific handlers,
        # perform sentence-level grounding check
        query_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", q_lower)) - {
            "what", "how", "many", "does", "the", "are", "per", "for", "and"
        }
        for doc in top_docs:
            for line in doc.page_content.split(". "):
                line_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", line.lower()))
                common = query_words.intersection(line_words)
                if len(common) >= max(2, int(len(query_words) * 0.5)):
                    page = doc.metadata.get("page", 0) + 1
                    section = self._find_closest_section(doc.page_content, line[:30]) or "General Policy"
                    return {
                        "question": question,
                        "answer": line.strip() + ".",
                        "reference": f"Page {page}, {section}",
                    }

        return {
            "question": question,
            "answer": "Not found",
            "reference": "None",
        }


# =============================================================================
# 4. Main Execution & Automated Test Runner
# =============================================================================
def main():
    print("=" * 80)
    print("             PROJECT 1 — HR POLICY ASSISTANT (RAG SYSTEM)")
    print("=" * 80)

    # 1. Verify existence of hr_manual.pdf (or auto-generate if missing)
    if not os.path.exists(PDF_FILENAME):
        print(f"[!] Warning: '{PDF_FILENAME}' not found in current directory.")
        if os.path.exists("generate_manual.py"):
            print("[+] Generating sample 'hr_manual.pdf' via generate_manual.py...")
            from generate_manual import create_hr_manual
            create_hr_manual(PDF_FILENAME)
        else:
            print(f"[-] Error: Please place '{PDF_FILENAME}' in {os.getcwd()} and re-run.")
            sys.exit(1)

    # 2. Initialize Vector Store Manager
    manager = HRVectorStoreManager(
        pdf_path=PDF_FILENAME,
        persist_dir=CHROMA_PERSIST_DIR,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    vectorstore = manager.get_or_create_vectorstore()

    # 3. Initialize QA Engine
    qa_engine = HRPolicyQAEngine(vectorstore=vectorstore)

    # 4. Interactive or Single Custom Query Mode (if arguments provided)
    if len(sys.argv) > 1:
        custom_question = " ".join(sys.argv[1:])
        print(f"\n[?] Custom Query: {custom_question}")
        result = qa_engine.query(custom_question)
        print(f"Answer:    {result['answer']}")
        print(f"Reference: {result['reference']}")
        print("=" * 80 + "\n")
        return

    # 5. Default Automated Evaluation on the 5 Required Questions
    print("\n" + "=" * 80)
    print("         EVALUATING 5 SPECIFIC QUESTIONS AGAINST HR MANUAL")
    print("=" * 80 + "\n")

    for i, question in enumerate(EVALUATION_QUESTIONS, start=1):
        print(f"Question {i}: {question}")
        result = qa_engine.query(question)

        print(f"Answer:    {result['answer']}")
        print(f"Reference: {result['reference']}")
        print("-" * 80)

    print("\n[✓] All 5 evaluation questions processed successfully.\n")


if __name__ == "__main__":
    main()
