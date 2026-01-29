import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF Chat – Hallucination Controlled", layout="wide")
st.title("📄 Chat with PDF (Quality Controlled RAG)")

SIMILARITY_THRESHOLD = 0.75

SYSTEM_PROMPT = """
You are an AI assistant.
Answer strictly using the provided document context.
If the answer is not present, reply exactly with:
❌ The answer is not available in the provided document.
Do not use external knowledge.
"""

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    query = st.text_input("Ask a question from the PDF")

    if query:
        scored = vectorstore.similarity_search_with_score(query, k=3)

        if not scored or scored[0][1] > SIMILARITY_THRESHOLD:
            st.warning("❌ The answer is not available in the provided document.")
            st.stop()

        context = "\n\n".join([d.page_content for d, _ in scored])

        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)
else:
    st.info("Upload a PDF to start.")
::contentReference[oaicite:0]{index=0}
