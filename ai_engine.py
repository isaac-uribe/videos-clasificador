import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Esto se carga al importar el archivo
print("Cargando modelo de IA (CLIP)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def obtener_embedding(frame_opencv):
    color_convertido = cv2.cvtColor(frame_opencv, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(color_convertido)

    inputs = processor(images=imagen_pil, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = model.get_image_features(**inputs)

    if hasattr(vision_outputs, 'pooler_output'):
        vector = vision_outputs.pooler_output
    else:
        vector = vision_outputs

    return vector.cpu().detach().numpy().flatten()