import os
import pickle

import cv2
import numpy as np
import ai_engine

BASE_PATH =r"C:\Users\isaac\media"

# 1. Cargar la memoria que generaste
with open("memoria_ia.pkl", "rb") as f:
    memoria = pickle.load(f)

# Extraemos los embeddings y las etiquetas a listas separadas para comparar rápido
conocidos_embeddings = np.array([item['embedding'] for item in memoria])
conocidos_etiquetas = [item['etiquetas'] for item in memoria]


def clasificar_video_nuevo(ruta_relativa):
    # Unir con tu constante BASE_PATH
    ruta_completa = os.path.join(BASE_PATH, ruta_relativa)

    cap = cv2.VideoCapture(ruta_completa)
    # Saltamos al segundo 2 para evitar pantallas negras
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Error al leer video"

    # Obtener el embedding del video nuevo
    nuevo_embedding = ai_engine.obtener_embedding(frame)

    # 2. Calcular la "distancia" (similitud) con todos los videos de la memoria
    # Usamos la distancia euclidiana (norma)
    distancias = np.linalg.norm(conocidos_embeddings - nuevo_embedding, axis=1)

    # 3. Obtener los índices de los 3 videos más parecidos
    indices_mas_cercanos = distancias.argsort()[:3]

    # 4. Recuperar las etiquetas de esos videos
    tags_sugeridos = []
    for i in indices_mas_cercanos:
        tags_sugeridos.append(conocidos_etiquetas[i])

    # Unimos todo y quitamos duplicados
    resultado = ", ".join(list(set(", ".join(tags_sugeridos).split(", "))))
    return resultado

if __name__ == "__main__":
    # Prueba con un video que NO esté en los 1,000
    video_test = r"C:\Users\isaac\media\video\f1fe392a-1607-4b31-95c0-08726bde5a31.mp4"
    tags = clasificar_video_nuevo(video_test)
    print(f"Tags sugeridos por la IA: {tags}")