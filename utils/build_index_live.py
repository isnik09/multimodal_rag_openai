import os
import faiss
import pickle
import numpy as np
from PIL import Image
import fitz
import docx
import torch
import open_clip
from sentence_transformers import SentenceTransformer

TEXT_DIR = "data/docs"
IMAGE_DIR = "data/images"
INDEX_DIR = "embeddings/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Load models
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model.to("cpu")

# Load existing index and metadata or initialize new
meta_path = os.path.join(INDEX_DIR, "meta.pkl")
index_path = os.path.join(INDEX_DIR, "index.faiss")

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(768 + 512)
    metadata = []

def read_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def embed_and_add_to_index(path):
    fname = os.path.basename(path)
    base = os.path.splitext(fname)[0]

    if fname.endswith(".pdf"):
        text = read_pdf(path)
    elif fname.endswith(".docx"):
        text = read_docx(path)
    elif fname.endswith(".txt"):
        text = read_text(path)
    else:
        text = None

    if text:
        text_emb = text_model.encode(text)
    else:
        text_emb = np.zeros(768)

    image_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(IMAGE_DIR, f"{base}{ext}")
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
        with torch.no_grad():
            image_emb = clip_model.encode_image(image).cpu().numpy()[0]
    else:
        image_emb = np.zeros(512)

    combined = np.concatenate([text_emb, image_emb])
    index.add(np.array([combined]).astype("float32"))
    metadata.append({
        "text": text[:1000] if text else "(No text)",
        "image_path": image_path
    })

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
