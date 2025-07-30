import os
import faiss
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
import open_clip
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
TEXT_DIR = "data/docs"
IMAGE_DIR = "data/images"
INDEX_DIR = "embeddings/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Load models
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model.to(device)

# Embedding storage
index_vectors = []
metadata = []

print("🔍 Reading and embedding documents...")

for filename in os.listdir(TEXT_DIR):
    if not filename.endswith(".txt"):
        continue
    path = os.path.join(TEXT_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Check for image with matching name
    basename = os.path.splitext(filename)[0]
    image_path = os.path.join(IMAGE_DIR, f"{basename}.jpg")
    if not os.path.exists(image_path):
        image_path = os.path.join(IMAGE_DIR, f"{basename}.png")

    # Embed text
    text_embedding = text_model.encode(text)

    # Embed image if it exists
    if os.path.exists(image_path):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(image).cpu().numpy()[0]
        combined = np.concatenate([text_embedding, image_embedding])
    else:
        combined = np.concatenate([text_embedding, np.zeros(512)])  # fill image slot

    index_vectors.append(combined)
    metadata.append({
        "text": text,
        "image_path": image_path if os.path.exists(image_path) else None
    })

# Save FAISS index
print("📦 Saving FAISS index and metadata...")
dim = len(index_vectors[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(index_vectors).astype("float32"))
faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

# Save metadata
with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("✅ Index built successfully.")
