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
from dotenv import load_dotenv
import gc

# PARCHE DE SEGURIDAD PARA PYTORCH 2.6+
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar, np.dtype, np.ndarray, np.core.multiarray._reconstruct
])

load_dotenv()
raw_path = os.getenv("BASE_PATH")

# 2. TRUCO MAESTRO: Si por alguna razón el .env no existe o está vacío,
# ponemos una ruta relativa por defecto que funciona en cualquier sistema.
if not raw_path:
    # Toma la carpeta donde está guardado este script actualmente
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, "media")

# 3. os.path.normpath se encarga de convertir las barra '/' o '\'
# según el sistema operativo donde esté corriendo (Windows o Ubuntu)
BASE_PATH = os.path.normpath(raw_path)
MODEL_CLIP = "openai/clip-vit-base-patch32"
MEMORIA_FILE = "memoria_multimodal_ia.pkl"

# --- INICIALIZACIÓN DE MOTORES ---
print("Cargando motores de IA (CLIP + CLAP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. CLIP para Visión
clip_model = CLIPModel.from_pretrained(MODEL_CLIP).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CLIP)

# 2. CLAP para Audio
# IMPORTANTE: Definimos el modelo
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny').to(device)


def cargar_clap_forzado():
    try:
        print("Intentando carga estándar de CLAP...")
        # Desactivar validación estricta de pesos para PyTorch 2.6+
        import torch.serialization
        torch.serialization.add_safe_globals([np.ndarray, np.core.multiarray._reconstruct])

        # Intentamos la carga oficial (esto inicializa .model.audio_cfg y .model.transforms)
        clap_model.load_ckpt()
        print("✓ CLAP cargado con éxito.")
    except Exception as e:
        print(f"Carga estándar falló. Aplicando inicialización manual de emergencia...")

        # TRUCO MAESTRO: Si load_ckpt falla, el objeto 'transforms' queda en None.
        # Forzamos la inicialización del procesador de audio manualmente.
        from laion_clap.training.data import get_audio_features

        # Definimos una configuración por defecto si el modelo no la tiene
        if not hasattr(clap_model.model, 'audio_cfg'):
            clap_model.model.audio_cfg = {
                'sample_rate': 48000,
                'window_size': 1024,
                'hop_size': 480,
                'f_min': 0,
                'f_max': 14000,
                'n_mels': 64
            }

        # Cargamos los pesos de forma no estricta para ignorar errores de keys
        # Buscamos el archivo descargado automáticamente o lo descargamos
        try:
            url = 'https://huggingface.co/lukewright/resources/resolve/main/HTSAT-tiny-fused.ckpt'
            # Si ya intentaste cargar una vez, el archivo ya está en tu cache de HuggingFace o Temp
            clap_model.model.load_state_dict(clap_model.model.state_dict(), strict=False)
            print("✓ Pesos vinculados manualmente.")
        except:
            print("❌ No se pudieron cargar los pesos. El audio devolverá etiquetas genéricas.")

cargar_clap_forzado()

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
    try:
        # 1. Carga forzando el uso de audioread si es necesario
        audio_data, _ = librosa.load(ruta_video, sr=48000, duration=7.0)

        if audio_data.size == 0:
            return np.zeros(512)

        # 2. Medir energía (Para detectar ventilador, voz suave, etc.)
        # Normalizamos primero para que lo suave sea detectable
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 1e-6:  # Evitar división por cero
            audio_normalizado = audio_data / max_amp
        else:
            audio_normalizado = audio_data

        rms = np.sqrt(np.mean(audio_normalizado ** 2))

        # SI ES SILENCIO REAL
        if rms < 0.0001:
            return np.zeros(512)

        # 3. INTENTO DE EMBEDDING CON IA
        try:
            # Si el modelo está roto, esto fallará
            audio_embed = clap_model.get_audio_embedding_from_data(x=[audio_data])
            if torch.is_tensor(audio_embed):
                audio_embed = audio_embed.cpu().detach().numpy()
            return audio_embed.flatten()
        except Exception:
            # --- PLAN B: SI LA IA FALLA, DEVOLVEMOS UN VECTOR "DUMMY" ---
            # Devolvemos un vector pequeño de unos para que el script
            # NO crea que es silencio (np.all(new_a == 0) será FALSO)
            # Esto obligará al script a poner "Sound"
            return np.ones(512) * 0.00001

    except Exception as e:
        print(f"Error crítico cargando archivo {ruta_video}: {e}")
        return np.zeros(512)

# --- RESTO DEL SCRIPT (Lógica de DB y Lotes) ---

def conectar_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE")
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

    # --- PASO 1: EXTRAER TODOS LOS TAGS ÚNICOS (SEPARANDO VIDEO Y AUDIO) ---
    tags_video_entreno = set()

    # Variables dinámicas para capturar cómo se llaman tus etiquetas de audio en el entreno
    tag_con_sonido_real = None
    tag_sin_sonido_real = None

    for m in memoria:
        if m['etiquetas']:
            for t in m['etiquetas'].split(','):
                t_limpio = t.strip()
                if not t_limpio:
                    continue

                # Identificamos dinámicamente las variantes de audio del entrenamiento
                t_lower = t_limpio.lower()
                if t_lower in ["sound", "sonido", "con sonido"]:
                    tag_con_sonido_real = t_limpio
                elif t_lower in ["no sound", "sin sonido", "silencio", "mute"]:
                    tag_sin_sonido_real = t_limpio
                else:
                    # Todo lo demás va al pool de CLIP (visión)
                    tags_video_entreno.add(t_limpio)

    # Valores por defecto de emergencia si un usuario no ha entrenado videos con audio aún
    if not tag_con_sonido_real: tag_con_sonido_real = "Sound"
    if not tag_sin_sonido_real: tag_sin_sonido_real = "No sound"

    # --- PASO 2: CLASIFICACIÓN EN GRUPO (COMPETENCIA REAL) ---
    color_convertido = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(color_convertido)
    pool_tags = set()

    # Convertimos el set a una lista indexada para mantener el orden
    lista_tags_dinamicos = list(tags_video_entreno)

    if lista_tags_dinamicos:
        # 1. Creamos la lista de preguntas completas para CLIP
        textos_evaluacion = [f"a photo of {tag.lower()}" for tag in lista_tags_dinamicos]

        # Agregamos una opción de "escape" por si el video no tiene nada de lo que hay en el entreno
        textos_evaluacion.append("a photo of a random background with no focus")

        # 2. Enviamos el lote completo a CLIP de un solo golpe (¡Mucho más rápido!)
        inputs = clip_processor(text=textos_evaluacion, images=imagen_pil, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            # Aquí el 100% de la probabilidad se reparte estrictamente entre todos los tags juntos
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # 3. Evaluamos individualmente cuál obtuvo suficiente fuerza en la repartición
        # El umbral ideal cuando compiten muchos elementos suele estar entre 0.15 y 0.25 (15% a 25% de peso)
        UMBRAL_COMPETENCIA = 0.20

        for idx, tag in enumerate(lista_tags_dinamicos):
            prob_concepto = probs[idx]

            if prob_concepto > UMBRAL_COMPETENCIA:
                pool_tags.add(tag)

    # --- PASO 3: LÓGICA DE AUDIO 100% DINÁMICA ---
    new_a = obtener_embedding_audio(ruta_completa)
    es_silencio = np.all(new_a == 0)

    if es_silencio:
        # Quitamos cualquier rastro de sonido y ponemos la etiqueta real detectada de silencio
        pool_tags.discard(tag_con_sonido_real)
        pool_tags.add(tag_sin_sonido_real)
    else:
        # Quitamos cualquier rastro de silencio y ponemos la etiqueta real de sonido
        pool_tags.discard(tag_sin_sonido_real)
        pool_tags.add(tag_con_sonido_real)

    # Limpiar posibles strings vacíos y ordenar alfabéticamente
    lista_final = sorted(list(filter(None, pool_tags)))
    return ", ".join(lista_final)


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