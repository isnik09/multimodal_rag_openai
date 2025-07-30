import os
import faiss
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
import open_clip
import fitz  # PyMuPDF
import docx

device = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_DIR = "data/docs"
IMAGE_DIR = "data/images"
INDEX_DIR = "embeddings/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model.to(device)

index_vectors = []
metadata = []

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def read_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif path.endswith(".docx"):
        return read_docx(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    return None

print("🔍 Reading and embedding documents...")

for filename in os.listdir(TEXT_DIR):
    full_path = os.path.join(TEXT_DIR, filename)
    text = read_file(full_path)

    if not text:
        print(f"⚠️ Skipping {filename} (unsupported or unreadable)")
        continue

    basename = os.path.splitext(filename)[0]
    image_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = os.path.join(IMAGE_DIR, f"{basename}{ext}")
        if os.path.exists(candidate):
            image_path = candidate
            break

    # Embed text
    text_embedding = text_model.encode(text)

    # Embed image if exists
    if image_path:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(image).cpu().numpy()[0]
        combined = np.concatenate([text_embedding, image_embedding])
    else:
        combined = np.concatenate([text_embedding, np.zeros(512)])  # fill image slot

    index_vectors.append(combined)
    metadata.append({
        "text": text[:1000],  # clip for performance
        "image_path": image_path
    })

# Save FAISS index
print("📦 Saving FAISS index and metadata...")
dim = len(index_vectors[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(index_vectors).astype("float32"))
faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("✅ Index built successfully.")
