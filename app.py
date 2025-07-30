import streamlit as st
from utils.embedder import embed_query
from utils.retriever import retrieve_relevant_chunks
from utils.generator import generate_answer

st.set_page_config(page_title="Multimodal RAG App", layout="wide")
st.title("🔍 Multimodal RAG with Streamlit")

query = st.text_input("Ask a question:")
if st.button("Search") and query:
    st.write("🔍 Retrieving relevant documents...")
    
    query_embedding = embed_query(query)
    results = retrieve_relevant_chunks(query_embedding, k=3)

    st.write("📄 Retrieved Contexts:")
    context = ""
    for item in results:
        st.image(item["image_path"], width=200)
        st.markdown(f"**Text:** {item['text']}")
        st.divider()
        context += item["text"] + "\n"

    st.write("🤖 Generating answer...")
    answer = generate_answer(query, context)
    st.success(answer)
