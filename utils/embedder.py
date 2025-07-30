from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def embed_text(text):
    return text_model.encode(text)

def embed_image(image_path):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(image).cpu().numpy()[0]

def embed_query(query):
    return embed_text(query)  # for now, just text
