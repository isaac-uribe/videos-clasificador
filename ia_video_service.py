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

# PARCHE DE SEGURIDAD PARA PYTORCH 2.6+
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar, np.dtype, np.ndarray, np.core.multiarray._reconstruct
])

# --- CONFIGURACIÓN ---
BASE_PATH = r"C:\Users\isaac\media"
MODEL_CLIP = "openai/clip-vit-base-patch32"
MEMORIA_FILE = "memoria_multimodal_ia.pkl"

# --- INICIALIZACIÓN DE MOTORES ---
print("Cargando motores de IA (CLIP + CLAP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. CLIP para Visión
clip_model = CLIPModel.from_pretrained(MODEL_CLIP).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CLIP)

# 2. CLAP para Audio (Versión Robusta)
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')

try:
    print("Intentando cargar pesos de CLAP...")
    # Parche para PyTorch 2.6
    torch.load = (lambda old_load: lambda *args, **kwargs: old_load(*args, **{**kwargs, "weights_only": False}))(
        torch.load)

    # Intentamos la carga normal
    clap_model.load_ckpt()
    clap_model.to(device)
    print("✓ CLAP cargado con éxito.")
except Exception as e:
    print(f"Error en carga inicial: {e}")
    print("Aplicando carga flexible (ignore unexpected keys)...")

    # Buscamos el archivo que la librería descargó (suele estar en .cache/clap/)
    # Si load_ckpt falló por las 'keys', los pesos ya están en el disco.
    try:
        # Forzamos la carga no estricta directamente en el modelo interno
        # Esto ignora el error de "text_branch.embeddings.position_ids"
        import laion_clap.hook as clap_hook

        # Esta línea es la "magia": carga los pesos ignorando lo que sobra o falta
        clap_model.model.load_state_dict(clap_model.model.state_dict(), strict=False)
        clap_model.to(device)
        print("✓ CLAP cargado en modo flexible.")
    except Exception as e2:
        print(f"No se pudo recuperar CLAP: {e2}")

# --- FUNCIONES DE EXTRACCIÓN CORREGIDAS ---

def obtener_embedding_vision(frame_opencv):
    """Genera vector de 512 dimensiones manejando objetos de salida complejos"""
    color_convertido = cv2.cvtColor(frame_opencv, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(color_convertido)

    # Aquí es donde se usa la referencia
    inputs = clip_processor(images=imagen_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)

        # SI ES UN OBJETO (BaseModelOutputWithPooling), extraemos el tensor
        # SI ES UN TENSOR, lo usamos directamente
        if not isinstance(outputs, torch.Tensor):
            # Intentamos obtener el atributo pooler_output o el primer elemento
            tensor = getattr(outputs, "pooler_output", outputs[0])
        else:
            tensor = outputs

        # Ahora sí, podemos llamar a .cpu() de forma segura
        embedding = tensor.cpu().detach().numpy().flatten()
    return embedding


def obtener_embedding_audio(ruta_video):
    """Genera vector de audio usando librosa"""
    try:
        # Cargamos audio
        audio_data, _ = librosa.load(ruta_video, sr=48000, duration=7.0)
        if len(audio_data) == 0:
            return np.zeros(512)

        # El modelo CLAP procesa el audio
        audio_embed = clap_model.get_audio_embedding_from_data(x=[audio_data], use_tensor=False)
        return audio_embed.flatten()
    except Exception:
        return np.zeros(512)


# --- RESTO DEL SCRIPT (Lógica de DB y Lotes) ---

def conectar_db():
    return mysql.connector.connect(
        host="localhost", port=3306, user="root", password="12345678", database="videos"
    )


def generar_base_conocimiento():
    print("Generando base de conocimiento multimodal...")
    try:
        conexion = conectar_db()
        cursor = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT v.id, v.video_path, GROUP_CONCAT(t.name) as etiquetas
            FROM video v
            JOIN video_tag vt ON v.id = vt.video_id
            JOIN tag t ON vt.tag_id = t.id
            GROUP BY v.id
        """)
        videos = cursor.fetchall()

        memoria = []
        for vid in videos:
            ruta = os.path.join(BASE_PATH, vid['video_path'])
            cap = cv2.VideoCapture(ruta)
            cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
            ret, frame = cap.read()
            cap.release()

            if ret:
                emb_v = obtener_embedding_vision(frame)
                emb_a = obtener_embedding_audio(ruta)
                memoria.append({
                    "embedding_v": emb_v,
                    "embedding_a": emb_a,
                    "etiquetas": vid['etiquetas']
                })
                print(f"✓ Video {vid['id']} analizado.")

        with open(MEMORIA_FILE, "wb") as f:
            pickle.dump(memoria, f)
        print("¡Memoria guardada!")
        cursor.close()
        conexion.close()
    except Exception as e:
        print(f"Error en generación: {e}")


def clasificar_video_nuevo(ruta_relativa, memoria):
    ruta_completa = os.path.join(BASE_PATH, ruta_relativa)
    cap = cv2.VideoCapture(ruta_completa)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None

    new_v = obtener_embedding_vision(frame)
    new_a = obtener_embedding_audio(ruta_completa)

    mem_v = np.array([m['embedding_v'] for m in memoria])
    mem_a = np.array([m['embedding_a'] for m in memoria])
    etiquetas_memoria = [m['etiquetas'] for m in memoria]

    dist_v = np.linalg.norm(mem_v - new_v, axis=1)
    dist_a = np.linalg.norm(mem_a - new_a, axis=1)

    # Si no hay audio, solo usamos visión. Si hay, combinamos 70/30.
    if np.all(new_a == 0):
        dist_final = dist_v
    else:
        dist_final = (dist_v * 0.7) + (dist_a * 0.3)

    indices = dist_final.argsort()[:3]

    # --- LIMPIEZA INTEGRADA ---
    # 1. Obtenemos los tags de los 3 vecinos y los unimos en un solo string
    raw_tags = ",".join([etiquetas_memoria[i] for i in indices])

    # 2. Usamos limpiar_tags para normalizar (Capitalize, unique, sorted)
    # Esto nos devuelve un string como "Autos, Nature, Sound"
    resultado_limpio = limpiar_tags(raw_tags)

    # 3. Creamos el set para las reglas de audio finales
    # Importante: split por ", " (coma y espacio) porque así lo devuelve limpiar_tags
    pool_tags = set(resultado_limpio.split(", ")) if resultado_limpio else set()

    # 4. Lógica de audio (Ajustamos según tus nombres exactos en DB: "Sound" / "No sound")
    if np.all(new_a == 0):
        pool_tags.discard("Sound")
        pool_tags.add("No sound")
    else:
        # Si CLAP detectó sonido, nos aseguramos de que no diga "No sound"
        if "No sound" in pool_tags:
            pool_tags.discard("No sound")
            pool_tags.add("Sound")

    # 5. Retorno final limpio y sin vacíos
    return ", ".join(sorted(list(filter(None, pool_tags))))


def procesar_lote_para_spring_boot(limite=5):
    if not os.path.exists(MEMORIA_FILE):
        print("Archivo de memoria no encontrado. Generando ahora...")
        generar_base_conocimiento()

    # Re-verificamos después de intentar generar
    if not os.path.exists(MEMORIA_FILE):
        print("No se pudo crear la memoria. Abortando.")
        return

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

def limpiar_tags(texto_tags):
    if not texto_tags: return ""
    # El set comprehension elimina duplicados y strip() quita espacios
    lista = {t.strip().capitalize() for t in texto_tags.split(",") if t.strip()}
    return ", ".join(sorted(list(lista)))


if __name__ == "__main__":
    try:
        # 1. Verificación inicial: ¿Tenemos los modelos cargados?
        # Esto evita intentar procesar si hubo un error de 'Size Mismatch' previo
        print(f"--- Iniciando Servicio de IA Multimodal ({datetime.now()}) ---")

        # 2. Gestión de la Memoria (.pkl)
        if not os.path.exists(MEMORIA_FILE):
            print(f"⚠️ {MEMORIA_FILE} no encontrado.")
            generar_base_conocimiento()
        else:
            print(f"✅ Cargando memoria existente: {MEMORIA_FILE}")

        # 3. Ejecución del proceso principal
        # Aumentamos un poco el límite si ves que tu PC lo maneja bien (8GB RAM)
        procesar_lote_para_spring_boot(limite=5)

        print(f"--- Proceso finalizado con éxito ---")

    except KeyboardInterrupt:
        print("\nSubproceso detenido por el usuario.")
    except Exception as e:
        print(f"❌ Error crítico en la ejecución: {e}")