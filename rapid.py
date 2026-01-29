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

st.set_page_config(page_title="Multi-Document PDF Chat", layout="wide")
st.title("📄 Multi-Document Reasoning with LLM (RAG)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

SIMILARITY_THRESHOLD = 0.75

SYSTEM_PROMPT = """
You are an AI assistant.
Answer strictly using the provided document context from multiple documents.
If the answer cannot be derived from the documents, respond exactly with:
❌ The answer is not available in the provided documents.
Do not use external knowledge.
"""

if uploaded_files:
    all_documents = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        all_documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    query = st.text_input("Ask a question across all uploaded PDFs")

    if query:
        results = vectorstore.similarity_search_with_score(query, k=5)

        if not results or results[0][1] > SIMILARITY_THRESHOLD:
            st.warning("❌ The answer is not available in the provided documents.")
            st.stop()

        context = "\n\n".join([doc.page_content for doc, _ in results])

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
    st.info("Upload one or more PDF files to begin.")
::contentReference[oaicite:0]{index=0}
