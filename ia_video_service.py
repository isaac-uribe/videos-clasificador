import os
import pickle
import cv2
import numpy as np
import mysql.connector
import torch
import librosa
import laion_clap
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime
import gc

# --- CONFIGURACIÓN ---
BASE_PATH = r"C:\Users\isaac\media"
MODEL_CLIP = "openai/clip-vit-base-patch32"
# Usamos la versión base de CLAP para balancear precisión y RAM
MODEL_CLAP_CKPT = "htsat-base"
MEMORIA_FILE = "memoria_multimodal_ia.pkl"

# --- INICIALIZACIÓN DE MOTORES ---
print("Cargando motores de IA (CLIP + CLAP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP para Visión
clip_model = CLIPModel.from_pretrained(MODEL_CLIP).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CLIP)

# CLAP para Audio
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')  # Versión tiny para ahorrar RAM
clap_model.load_ckpt()
clap_model.to(device)


# --- FUNCIONES DE EXTRACCIÓN ---

def obtener_embedding_vision(frame_opencv):
    """Genera vector de 512 dimensiones de la imagen"""
    color_convertido = cv2.cvtColor(frame_opencv, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(color_convertido)
    inputs = clip_processor(images=imagen_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = clip_model.get_image_features(**inputs)
    return vision_outputs.cpu().detach().numpy().flatten()


def obtener_embedding_audio(ruta_video):
    """Genera vector de 512 dimensiones del sonido"""
    try:
        # Cargamos solo 7 segundos de audio para no saturar RAM
        audio_data, _ = librosa.load(ruta_video, sr=48000, duration=7.0)
        if len(audio_data) == 0:
            return np.zeros(512)

        # El modelo CLAP espera audio en formato float32
        audio_embed = clap_model.get_audio_embedding_from_data(x=[audio_data], use_tensor=False)
        return audio_embed.flatten()
    except Exception:
        # Si el video no tiene pista de audio, devolvemos un vector de ceros
        return np.zeros(512)


def conectar_db():
    return mysql.connector.connect(
        host="localhost", port=3306, user="root", password="12345678", database="videos"
    )


def limpiar_tags(texto_tags):
    if not texto_tags: return ""
    lista = [t.strip() for t in texto_tags.split(",") if t.strip()]
    return ", ".join(sorted(list(set(lista))))


# --- LÓGICA DE PROCESAMIENTO ---

def generar_base_conocimiento():
    """Crea la memoria con embeddings de video y audio"""
    print("Generando base de conocimiento multimodal...")
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

        memoria_multimodal = []
        for vid in videos:
            ruta_completa = os.path.join(BASE_PATH, vid['video_path'])

            # 1. Procesar Imagen
            cap = cv2.VideoCapture(ruta_completa)
            cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
            ret, frame = cap.read()
            cap.release()

            if ret:
                emb_v = obtener_embedding_vision(frame)
                # 2. Procesar Audio
                emb_a = obtener_embedding_audio(ruta_completa)

                tags = limpiar_tags(vid['etiquetas'])

                memoria_multimodal.append({
                    "embedding_v": emb_v,
                    "embedding_a": emb_a,
                    "etiquetas": tags
                })
                print(f"✓ ID {vid['id']} procesado (Vista + Audio).")

            # Limpieza de RAM manual cada cierto tiempo
            if len(memoria_multimodal) % 10 == 0:
                gc.collect()

        with open(MEMORIA_FILE, "wb") as f:
            pickle.dump(memoria_multimodal, f)

        cursor.close()
        conexion.close()
        print("¡Memoria Multimodal creada!")
    except Exception as e:
        print(f"Error: {e}")


def clasificar_video_nuevo(ruta_relativa, memoria):
    """Compara el video nuevo usando visión y audio simultáneamente"""
    ruta_completa = os.path.join(BASE_PATH, ruta_relativa)

    cap = cv2.VideoCapture(ruta_completa)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None

    # Embeddings del nuevo video
    new_v = obtener_embedding_vision(frame)
    new_a = obtener_embedding_audio(ruta_completa)

    # Extraer arrays de la memoria
    mem_v = np.array([m['embedding_v'] for m in memoria])
    mem_a = np.array([m['embedding_a'] for m in memoria])
    etiquetas = [m['etiquetas'] for m in memoria]

    # Calcular distancias (70% peso a la imagen, 30% al audio para balancear)
    dist_v = np.linalg.norm(mem_v - new_v, axis=1)
    dist_a = np.linalg.norm(mem_a - new_a, axis=1)

    # Si el video nuevo es mudo (vector de ceros), ignoramos la distancia de audio
    if np.all(new_a == 0):
        dist_final = dist_v
    else:
        dist_final = (dist_v * 0.7) + (dist_a * 0.3)

    indices = dist_final.argsort()[:3]

    pool = []
    for i in indices:
        pool.extend(etiquetas[i].split(", "))

    # Lógica de audio forzada: si no hay audio real detectado por CLAP, quitamos 'Sound'
    resultado = sorted(list(set(pool)))
    if np.all(new_a == 0) and "Sound" in resultado:
        resultado.remove("Sound")
        if "No Sound" not in resultado: resultado.append("No Sound")

    return ", ".join(resultado)


def procesar_lote_para_spring_boot(limite=5):
    if not os.path.exists(MEMORIA_FILE):
        print("Falta memoria multimodal. Generando...")
        generar_base_conocimiento()

    with open(MEMORIA_FILE, "rb") as f:
        memoria = pickle.load(f)

    try:
        conexion = conectar_db()
        cursor = conexion.cursor(dictionary=True)

        query = """
            SELECT v.id, v.video_path FROM video v
            LEFT JOIN video_tag vt ON v.id = vt.video_id
            LEFT JOIN video_tag_temporal vtt ON v.id = vtt.video_id
            WHERE vt.video_id IS NULL AND vtt.video_id IS NULL LIMIT %s
        """
        cursor.execute(query, (limite,))
        pendientes = cursor.fetchall()

        for vid in pendientes:
            tags = clasificar_video_nuevo(vid['video_path'], memoria)
            if tags:
                cursor.execute("""
                    INSERT INTO video_tag_temporal (video_id, tags_suggest, confirm, date_creation)
                    VALUES (%s, %s, %s, %s)
                """, (vid['id'], tags, False, datetime.now()))
                conexion.commit()
                print(f"✓ Video {vid['id']} clasificado: {tags}")

        cursor.close()
        conexion.close()
    except Exception as e:
        print(f"Error en lote: {e}")


if __name__ == "__main__":
    # La primera vez DEBES generar la base multimodal
    # generar_base_conocimiento()
    procesar_lote_para_spring_boot(limite=5)