import os
import shutil
import streamlit as st
from utils.embedder import embed_query
from utils.retriever import retrieve_relevant_chunks
from utils.generator import generate_answer
from utils.build_index_live import embed_and_add_to_index

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("📚 Multimodal RAG System")

# Paths
DOCS_DIR = "data/docs"
IMAGES_DIR = "data/images"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# === File Upload Section ===
st.subheader("📂 Upload New Document or Image")

uploaded_files = st.file_uploader(
    "Upload a .pdf, .docx, .txt or image file (.jpg/.png)", 
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if st.button("📌 Generate Embeddings from Uploaded Files"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        for file in uploaded_files:
            filename = file.name
            file_path = os.path.join(DOCS_DIR if filename.endswith((".txt", ".pdf", ".docx")) else IMAGES_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(file.read())
            embed_and_add_to_index(file_path)
        st.success("✅ Embedding(s) generated and added to index!")

st.divider()

# === Query Section ===
st.subheader("🔍 Ask a Question")

query = st.text_input("Enter your question here:")
model_choice = st.selectbox("Choose model", ["gpt-3.5-turbo", "gpt-4"])

if st.button("🧠 Search & Answer"):
    if not query:
        st.warning("Enter a question first.")
    else:
        query_embedding = embed_query(query)
        results = retrieve_relevant_chunks(query_embedding, k=3)

        context = ""
        for res in results:
            if res.get("image_path"):
                st.image(res["image_path"], width=200)
            st.markdown(f"**Text:** {res['text'][:500]}")
            st.divider()
            context += res["text"] + "\n"

        st.subheader("🧠 Answer")
        answer = generate_answer(query, context, model=model_choice)
        st.success(answer)
