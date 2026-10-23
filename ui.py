#!/usr/bin/env python3
"""
=============================================================================
Project 1 — HR Policy Assistant (Interactive Web UI)
=============================================================================
Interactive Web Application built with:
  - Gradio for the responsive, modern UI
  - ChromaDB for local vector similarity retrieval
  - HuggingFaceEmbeddings ("all-MiniLM-L6-v2")
  - LangChain for document loading and vector store management

Key Features:
  1. Loads existing vector database from './chroma_db' or indexes 'hr_manual.pdf'.
  2. Offers both a dedicated Q&A interface and an interactive Chat Assistant.
  3. Preloaded with the 5 project evaluation questions as one-click examples.
  4. Answers strictly based on the document with exact page/section references.
  5. Outputs strictly "Not found" when information is absent from the manual.
  6. Displays retrieved source context chunks with relevance metrics for full transparency.
=============================================================================
"""

import os
import sys
import warnings
from typing import List, Dict, Any, Tuple

# Suppress deprecation and parallelism warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
from app import (
    HRVectorStoreManager,
    HRPolicyQAEngine,
    PDF_FILENAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    EVALUATION_QUESTIONS,
)


# =============================================================================
# 1. System Initialization
# =============================================================================
print("[+] Initializing HR Policy Assistant for Web UI...")

# Ensure PDF and ChromaDB are available
if not os.path.exists(PDF_FILENAME):
    if os.path.exists("generate_manual.py"):
        print(f"[+] '{PDF_FILENAME}' not found. Generating reference manual...")
        from generate_manual import create_hr_manual
        create_hr_manual(PDF_FILENAME)

manager = HRVectorStoreManager(
    pdf_path=PDF_FILENAME,
    persist_dir=CHROMA_PERSIST_DIR,
    embedding_model_name=EMBEDDING_MODEL_NAME,
)
vectorstore = manager.get_or_create_vectorstore()
qa_engine = HRPolicyQAEngine(vectorstore=vectorstore)
print("[✓] RAG Engine ready. Launching Gradio interface...\n")


# =============================================================================
# 2. Core Q&A and Chat Handlers
# =============================================================================
def handle_qa_query(question: str) -> Tuple[str, str, str, str]:
    """
    Handles a query submitted via the Q&A tab.
    Returns:
      - Formatted Answer Markdown
      - Reference String
      - Status Badge HTML
      - Source Context / Retrieved Chunks text
    """
    if not question or not question.strip():
        return (
            "⚠️ *Please enter a valid question.*",
            "None",
            '<span style="background-color: #E2E8F0; color: #4A5568; padding: 4px 10px; border-radius: 12px; font-weight: bold;">Empty Query</span>',
            "No query entered.",
        )

    result = qa_engine.query(question.strip())
    answer = result.get("answer", "Not found")
    reference = result.get("reference", "None")

    # Format Status Badge
    if answer == "Not found":
        status_html = (
            '<span style="background-color: #FED7D7; color: #C53030; padding: 5px 12px; border-radius: 14px; font-weight: 600; font-size: 0.9em;">'
            '❌ Not Found in HR Manual</span>'
        )
        formatted_answer = (
            "### **Answer:**\n"
            "> **Not found**\n\n"
            "*The requested information is not present in the HR Policy Manual, or the question is outside the scope of company policies.*"
        )
    else:
        status_html = (
            '<span style="background-color: #C6F6D5; color: #22543D; padding: 5px 12px; border-radius: 14px; font-weight: 600; font-size: 0.9em;">'
            '✓ Grounded in Document</span>'
        )
        formatted_answer = (
            f"### **Answer:**\n"
            f"{answer}"
        )

    # Format Retrieved Chunks for Transparency
    retrieved_results = vectorstore.similarity_search_with_score(question, k=3)
    chunks_display = []
    for idx, (doc, dist) in enumerate(retrieved_results, start=1):
        page_num = doc.metadata.get("page", 0) + 1
        preview = doc.page_content.strip()
        chunks_display.append(
            f"--- [Chunk {idx} | Page {page_num} | Distance: {dist:.4f}] ---\n{preview}"
        )
    context_str = "\n\n".join(chunks_display)

    return formatted_answer, reference, status_html, context_str


def handle_chat_interaction(message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    """
    Handles conversational interactions in the Chatbot tab.
    Maintains user/assistant message history dictionary format.
    """
    if not message or not message.strip():
        return history, ""

    result = qa_engine.query(message.strip())
    answer = result.get("answer", "Not found")
    reference = result.get("reference", "None")

    if answer == "Not found":
        bot_response = (
            "**Answer:** Not found\n\n"
            "*Reference:* None (This topic is not covered in the HR Policy Manual)."
        )
    else:
        bot_response = (
            f"**Answer:** {answer}\n\n"
            f"📌 **Reference:** `{reference}`"
        )

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_response},
    ]
    return updated_history, ""


# =============================================================================
# 3. Gradio Interface Definition
# =============================================================================
custom_css = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.header-box h1 {
    color: #FFFFFF !important;
    font-size: 26px !important;
    margin-bottom: 6px !important;
}
.header-box p {
    color: #E2E8F0 !important;
    font-size: 14px !important;
    margin: 0 !important;
}
.ref-box textarea {
    font-family: monospace !important;
    font-weight: 600 !important;
    color: #2B6CB0 !important;
}
"""

with gr.Blocks(title="HR Policy Assistant — Project 1") as demo:

    # Header Banner
    gr.HTML(
        """
        <div class="header-box">
            <h1>🏢 Project 1: HR Policy Assistant</h1>
            <p>Local Retrieval-Augmented Generation (RAG) System powered by LangChain, ChromaDB, and HuggingFace Embeddings (<code>all-MiniLM-L6-v2</code>).</p>
        </div>
        """
    )

    with gr.Tabs():
        # TAB 1: Structured Q&A
        with gr.TabItem("🔍 Policy Q&A & Citations", id="qa_tab"):
            with gr.Row():
                with gr.Column(scale=5):
                    query_input = gr.Textbox(
                        label="Ask an HR Policy Question",
                        placeholder="Type any question about leave, probation, medical insurance, conduct, etc...",
                        lines=2,
                        autofocus=True,
                    )

                    with gr.Row():
                        submit_btn = gr.Button("🔎 Search Policy", variant="primary", scale=2)
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                    # Clickable Examples from Project 1 Specification
                    gr.Markdown("#### 💡 Quick Evaluation Questions (Click to Test):")
                    example_list = [[q] for q in EVALUATION_QUESTIONS] + [
                        ["What is the notice period during probation?"],
                        ["What is the standard core working hours and remote work policy?"],
                    ]
                    gr.Examples(
                        examples=example_list,
                        inputs=query_input,
                        label=None,
                    )

                with gr.Column(scale=5):
                    status_badge = gr.HTML(
                        value='<span style="background-color: #E2E8F0; color: #4A5568; padding: 5px 12px; border-radius: 14px; font-weight: 600; font-size: 0.9em;">Ready</span>'
                    )
                    answer_output = gr.Markdown(
                        value="*Submit a question above to retrieve the answer from the HR manual.*"
                    )
                    reference_output = gr.Textbox(
                        label="Exact Reference Citation",
                        interactive=False,
                        elem_classes=["ref-box"],
                    )

            with gr.Accordion("📄 Retrieved Source Chunks & Relevance Distance", open=False):
                context_output = gr.Textbox(
                    label="Underlying Vector Search Context Chunks",
                    lines=8,
                    interactive=False,
                )

            # Wiring up interactions
            submit_btn.click(
                fn=handle_qa_query,
                inputs=query_input,
                outputs=[answer_output, reference_output, status_badge, context_output],
            )
            query_input.submit(
                fn=handle_qa_query,
                inputs=query_input,
                outputs=[answer_output, reference_output, status_badge, context_output],
            )
            clear_btn.click(
                fn=lambda: ("", "*Submit a question above to retrieve the answer from the HR manual.*", "", '<span style="background-color: #E2E8F0; color: #4A5568; padding: 5px 12px; border-radius: 14px; font-weight: 600; font-size: 0.9em;">Ready</span>', ""),
                inputs=None,
                outputs=[query_input, answer_output, reference_output, status_badge, context_output],
            )

        # TAB 2: Conversational Chat Assistant
        with gr.TabItem("💬 Conversational Assistant", id="chat_tab"):
            chatbot = gr.Chatbot(
                label="HR Policy Chatbot",
                height=420,
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a question conversationally (e.g., How many days of maternity leave are given?)...",
                    show_label=False,
                    scale=5,
                )
                chat_send_btn = gr.Button("Send", variant="primary", scale=1)
                chat_clear_btn = gr.Button("Reset Chat", scale=1)

            chat_send_btn.click(
                fn=handle_chat_interaction,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=handle_chat_interaction,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_clear_btn.click(
                fn=lambda: ([], ""),
                inputs=None,
                outputs=[chatbot, chat_input],
            )

        # TAB 3: HR Manual Overview & Summary
        with gr.TabItem("📖 Document Overview", id="doc_tab"):
            gr.Markdown(
                """
                ### **Active Document:** `hr_manual.pdf`
                - **Document Type:** Acme Corp Employee Regulations & Operational Policies
                - **Vector Database:** Local ChromaDB (`./chroma_db`)
                - **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 Dimensions)
                - **Chunking Strategy:** `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`

                ---
                #### **Document Sections Covered:**
                1. **Section 1:** Employment Classification and Scope
                2. **Section 2:** Probation and Confirmation Policy
                   - *Section 2.2:* Duration of Probation (6 months up to Senior Manager; 12 months for Director & above)
                   - *Section 2.3:* Notice Period During Probation (15 calendar days)
                3. **Section 3:** Leave and Absence Administration
                   - *Section 3.1:* Privilege Leave (PL) — 18 days per calendar year (accrual at 1.5 days/month)
                   - *Section 3.2:* Casual Leave (7 days) & Sick Leave (7 days)
                   - *Section 3.3:* Maternity Leave — 26 weeks (182 days) paid leave; Paternity Leave (5 days)
                4. **Section 4:** Employee Benefits and Insurance Coverage
                   - *Section 4.1:* Group Medical Health Insurance — INR 5,00,000 cover; Company pays INR 15,000 annual premium (0 employee contribution)
                   - *Section 4.2:* Group Term Life Insurance — 3x Annual Fixed CTC
                5. **Section 5:** General Code of Conduct and Disciplinary Norms
                   - *Section 5.1:* Working Hours (9:30 AM to 6:00 PM IST) & Hybrid Policy (up to 2 WFH days/week)
                   - *Section 5.2:* Separation Notice Period Post Confirmation (60 calendar days)
                """
            )

    gr.Markdown(
        """
        ---
        <center>
            <small style="color: #718096;">
                Project 1 — HR Policy Assistant | Built with LangChain, ChromaDB & Gradio
            </small>
        </center>
        """
    )


# =============================================================================
# 4. Entrypoint
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    print(f"[+] Launching Gradio Web UI on http://127.0.0.1:{port}...")
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
