import streamlit as st
from utils.embedder import embed_query
from utils.retriever import retrieve_relevant_chunks
from utils.generator import generate_answer

st.set_page_config(page_title="🧠 Multimodal RAG", layout="wide")
st.title("🔍 Multimodal Retrieval-Augmented Generation")

query = st.text_input("Ask a question:")
model_choice = st.selectbox("Choose model", ["gpt-3.5-turbo", "gpt-4"])

if st.button("Search") and query:
    st.write("🔎 Retrieving documents...")
    query_embedding = embed_query(query)
    results = retrieve_relevant_chunks(query_embedding, k=3)

    context = ""
    for res in results:
        if res.get("image_path"):
            st.image(res["image_path"], width=200)
        st.markdown(f"**Text:** {res['text'][:500]}")  # Preview
        st.divider()
        context += res["text"] + "\n"

    st.write("🤖 Generating answer...")
    answer = generate_answer(query, context, model=model_choice)
    st.success(answer)
