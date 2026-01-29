# LLM
# 📄 Chat with PDF using LLM & RAG

## 📌 Overview
This project is a **Chat with PDF application** built using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)**.  
It allows users to upload a PDF file and ask questions in natural language. The system retrieves relevant content from the document and generates accurate, context-aware answers.

---

## 🚀 Features
- Upload and process PDF documents
- Ask natural language questions from the PDF
- Semantic search using embeddings
- Accurate, document-based answers (RAG)
- Simple and interactive UI using Streamlit

---

## 🧠 System Architecture

PDF Upload
↓
Text Extraction (PyPDFLoader)
↓
Chunking (Overlapping Chunks)
↓
Embeddings Generation
↓
Vector Store (FAISS)
↓
User Query
↓
Relevant Chunk Retrieval
↓
LLM Answer Generation


---

## 🛠️ Technology Stack

| Component | Technology |
|----------|------------|
| Programming Language | Python 3.12 |
| UI Framework | Streamlit |
| LLM | OpenAI GPT-3.5 Turbo |
| Embeddings | OpenAI Embeddings |
| Vector Database | FAISS |
| RAG Framework | LangChain |
| PDF Loader | PyPDFLoader |
| Environment Variables | python-dotenv |

---

## 📂 Project Structure

- pdf.py # Main Streamlit application
- README.md # Project documentation
- .env # OpenAI API key (ignored in GitHub)


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone htps://github.com/your-username/chat-with-pdf-rag.git
cd chat-with-pdf-rag
