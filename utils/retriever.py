# utils/retriever.py

import faiss
import os
import pickle
import numpy as np

# Load metadata and FAISS index
with open("embeddings/faiss_index/meta.pkl", "rb") as f:
    metadata = pickle.load(f)

index = faiss.read_index("embeddings/faiss_index/index.faiss")

def retrieve_relevant_chunks(query_embedding, k=3):
    query_vector = np.array([query_embedding]).astype('float32')
    D, I = index.search(query_vector, k)
    return [metadata[i] for i in I[0]]
