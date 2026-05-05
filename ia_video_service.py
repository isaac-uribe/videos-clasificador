import os
import pickle
import cv2
import numpy as np
import mysql.connector
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime

# --- CONFIGURACIÓN Y CONSTANTES ---
BASE_PATH = r"C:\Users\isaac\media"
MODEL_NAME = "openai/clip-vit-base-patch32"
MEMORIA_FILE = "memoria_ia.pkl"

# --- MOTOR DE IA (AI ENGINE CENTRALIZADO) ---
print("Cargando modelo de IA (CLIP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def obtener_embedding(frame_opencv):
    """Convierte un frame de video en un vector numérico (embedding)"""
    color_convertido = cv2.cvtColor(frame_opencv, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(color_convertido)

    inputs = processor(images=imagen_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = model.get_image_features(**inputs)

    # Manejo de la salida del modelo para asegurar el tensor correcto
    vector = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs
    return vector.cpu().detach().numpy().flatten()

# --- UTILIDADES DE BASE DE DATOS ---
def conectar_db():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="12345678",
        database="videos"
    )

# --- LÓGICA DE NEGOCIO ---

def generar_base_conocimiento():
    """Entrena la 'memoria' usando los 1000 videos ya etiquetados"""
    print("Generando base de conocimiento desde MySQL...")
    try:
        conexion = conectar_db()
        cursor = conexion.cursor(dictionary=True)

        query = """
        SELECT v.id, v.video_path, GROUP_CONCAT(t.name) as etiquetas
        FROM video v
        JOIN video_tag vt ON v.id = vt.video_id
        JOIN tag t ON vt.tag_id = t.id
        GROUP BY v.id
        """
        cursor.execute(query)
        videos = cursor.fetchall()

        datos_entrenamiento = []
        for vid in videos:
            ruta_completa = os.path.join(BASE_PATH, vid['video_path'])
            cap = cv2.VideoCapture(ruta_completa)
            cap.set(cv2.CAP_PROP_POS_MSEC, 2000)  # Saltar 2 seg
            ret, frame = cap.read()

            if ret:
                embedding = obtener_embedding(frame)
                datos_entrenamiento.append({
                    "embedding": embedding,
                    "etiquetas": vid['etiquetas']
                })
                print(f"✓ Video {vid['id']} procesado.")
            cap.release()

        with open(MEMORIA_FILE, "wb") as f:
            pickle.dump(datos_entrenamiento, f)

        cursor.close()
        conexion.close()
        print("¡Memoria de IA creada y guardada con éxito!")

    except Exception as e:
        print(f"Error en generación de memoria: {e}")


def clasificar_video_nuevo(ruta_relativa, conocidos_embeddings, conocidos_etiquetas):
    """Predice tags para un solo video basado en la memoria cargada"""
    ruta_completa = os.path.join(BASE_PATH, ruta_relativa)
    cap = cv2.VideoCapture(ruta_completa)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    nuevo_embedding = obtener_embedding(frame)
    # Cálculo de distancia euclidiana
    distancias = np.linalg.norm(conocidos_embeddings - nuevo_embedding, axis=1)
    indices_mas_cercanos = distancias.argsort()[:3]

    tags_sugeridos = []
    for i in indices_mas_cercanos:
        tags_sugeridos.append(conocidos_etiquetas[i])

    # Limpieza de duplicados y formato string
    resultado = ", ".join(list(set(", ".join(tags_sugeridos).split(", "))))
    return resultado


def procesar_lote_para_spring_boot(limite=5):
    """Busca videos sin tags e inserta sugerencias en la tabla temporal"""
    if not os.path.exists(MEMORIA_FILE):
        print("Error: No existe el archivo de memoria. Ejecuta primero generar_base_conocimiento()")
        return

    # Cargar memoria en RAM
    with open(MEMORIA_FILE, "rb") as f:
        memoria = pickle.load(f)

    conocidos_embeddings = np.array([item['embedding'] for item in memoria])
    conocidos_etiquetas = [item['etiquetas'] for item in memoria]

    try:
        conexion = conectar_db()
        cursor = conexion.cursor(dictionary=True)

        # Buscar videos que no están en video_tag ni en video_tag_temporal
        query_pendientes = """
            SELECT v.id, v.video_path 
            FROM video v
            LEFT JOIN video_tag vt ON v.id = vt.video_id
            LEFT JOIN video_tag_temporal vtt ON v.id = vtt.video_id
            WHERE vt.video_id IS NULL AND vtt.video_id IS NULL
            LIMIT %s
        """
        cursor.execute(query_pendientes, (limite,))
        videos_pendientes = cursor.fetchall()

        for vid in videos_pendientes:
            tags = clasificar_video_nuevo(vid['video_path'], conocidos_embeddings, conocidos_etiquetas)

            if tags:
                # Insertar en la tabla que creaste para Spring Boot
                query_insert = """
                    INSERT INTO video_tag_temporal (video_id, tags_suggest, confirm, date_creation)
                    VALUES (%s, %s, %s, %s)
                """
                valores = (vid['id'], tags, False, datetime.now())
                cursor.execute(query_insert, valores)
                conexion.commit()
                print(f"✓ Video ID {vid['id']} clasificado: {tags}")

        cursor.close()
        conexion.close()
        print("--- Lote completado ---")

    except Exception as e:
        print(f"Error en procesamiento de lote: {e}")


# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    # 1. Si no tienes el .pkl, descomenta la siguiente línea una vez:
    # generar_base_conocimiento()

    # 2. Ejecutar el proceso para la tabla temporal
    procesar_lote_para_spring_boot(limite=5)