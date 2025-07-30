
import torch
from PIL import Image
import open_clip
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Text model
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model.to(device)

def embed_text(text):
    return text_model.encode(text)

def embed_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features.cpu().numpy()[0]

def embed_query(query):
    return embed_text(query)
