import os.path
import pickle

import cv2
import mysql.connector
import ai_engine

BASE_PATH =r"C:\Users\isaac\media"

def conectar_db():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="12345678",
        database="videos"
    )


def extraer_frame_prueba(ruta_video):

    ruta_completa = os.path.join(BASE_PATH, ruta_video)

    print(f"Intentando abrir: {ruta_completa}")

    cap = cv2.VideoCapture(ruta_completa)
    if not cap.isOpened():
        print("no pude abrir el video")
        return

    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_frame.jpg", frame)
        print("Frame extraido con exito")

        cap.release()


def procesar_lote_videos():
    try:
        conexion = conectar_db()

        if conexion.is_connected():
            cursor = conexion.cursor(dictionary=True)

            cursor.execute("SELECT id, video_path FROM video LIMIT 10")
            lista_videos = cursor.fetchall()

            for vid in lista_videos:
                print(f"Analisando ID {vid['id']}....")

                extraer_frame_prueba(vid['video_path'])

                cursor.close()
                conexion.close()
    except mysql.connector.Error as e:
        print(f"Error de base de datos: {e}")

def generar_base_conocimiento():
    try:
        conexion = conectar_db()

        if conexion.is_connected():
            cursor = conexion.cursor(dictionary=True)

            query = """
            SELECT v.id, v.video_path, GROUP_CONCAT(t.name) as etiquetas
            FROM video v
            JOIN video_tag vt ON v.id = vt.video_id
            JOIN tag t ON vt.tag_id = t.id
            GROUP BY v.id
            LIMIT 1000
            """

            cursor.execute(query)
            videos = cursor.fetchall()

            datos_entrenamiento = []

            for vid in videos:
                ruta_completa = os.path.join(BASE_PATH, vid['video_path'])

                cap = cv2.VideoCapture(ruta_completa)
                ret, frame = cap.read()
                if ret:
                    embedding = ai_engine.obtener_embedding(frame)

                    datos_entrenamiento.append({
                        "embedding": embedding,
                        "etiquetas": vid['etiquetas']
                    })
                    print(f"Video {vid['id']} procesado.")
                cap.release()

            with open("memoria_ia.pkl", "wb") as f:
                pickle.dump(datos_entrenamiento, f)

            print("¡Memoria de IA creada con éxito!")
    except mysql.connector.Error as e:
        print(f"Error de base de datos: {e}")

if __name__ == "__main__":
    generar_base_conocimiento()