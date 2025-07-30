from utils.retriever import retrieve_relevant_chunks
from utils.embedder import embed_query, embed_and_add_to_index
from utils.generator import generate_answer
import os

# User uploads files and clicks "Generate Embeddings"
uploaded_files = st.file_uploader("📄 Upload PDF, DOCX or Images", accept_multiple_files=True)

if st.button("⚙️ Generate Embeddings"):
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join("data/docs", file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        embed_and_add_to_index("data/docs")
        st.success("✅ Embeddings generated!")
    else:
        st.warning("📎 Please upload a file first.")

# Once embeddings exist, allow user to ask questions
if os.path.exists("embeddings/faiss_index/index.faiss"):
    query = st.text_input("🔍 Ask a question:")
    if st.button("🔎 Search") and query:
        query_embedding = embed_query(query)
        try:
            results = retrieve_relevant_chunks(query_embedding, k=3)
            context = "\n".join(r["text"] for r in results)
            st.write("📚 Retrieved context:")
            for r in results:
                if r.get("image_path"):
                    st.image(r["image_path"], width=200)
                st.markdown(f"**Text:** {r['text'][:300]}")
            st.divider()

            answer = generate_answer(query, context)
            st.success(answer)
        except ValueError as e:
            st.error(str(e))
else:
    st.info("📂 No FAISS index found. Please upload files and generate embeddings first.")
