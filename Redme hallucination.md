# 📄 Chat with PDF – Hallucination & Quality Controlled RAG

## 📌 Project Overview
This project is an **LLM-powered Chat with PDF application** enhanced with **hallucination control and quality guardrails**.  
Users can upload a PDF and ask questions, and the system ensures that answers are **strictly grounded in the document content**.  
If the information is not present in the PDF, the system clearly states that instead of guessing.

This prototype addresses **LLM hallucination issues** using **confidence thresholds, prompt constraints, and source grounding**.

---

## 🚀 Key Features
- Upload and analyze PDF documents
- Ask natural language questions
- Retrieval-Augmented Generation (RAG)
- Hallucination control with confidence checks
- Explicit “Not available in document” responses
- Simple UI built with Streamlit

---

## 🧠 Hallucination Control Strategy

### Implemented Guardrails
1. **Source-Grounded Prompt Constraint**  
   The model is instructed to answer strictly from the retrieved document context.

2. **Confidence Threshold (Similarity Score Check)**  
   If retrieved chunks do not meet a similarity threshold, the system blocks the response.

3. **Explicit Fallback Response**  
   When the answer is not found, the system responds with:  
   `❌ The answer is not available in the provided document.`

---


