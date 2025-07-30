import os
import pickle
import faiss
import numpy as np

INDEX_PATH = "embeddings/faiss_index/index.faiss"
META_PATH = "embeddings/faiss_index/meta.pkl"

def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve_relevant_chunks(query_embedding, k=3):
    index, metadata = load_index_and_metadata()
    if index is None or metadata is None:
        raise ValueError("Index not found. Please generate embeddings first.")

    _, indices = index.search(np.array([query_embedding]), k)
    results = [metadata[i] for i in indices[0] if i < len(metadata)]
    return results
