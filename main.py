from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
import os
import uvicorn

app = FastAPI()

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Función para determinar el gesto de la mano
def detectar_gesto(mano_landmarks):
    dedos_arriba = []
    dedos_indices = [8, 12, 16, 20]

    for i in dedos_indices:
        if mano_landmarks.landmark[i].y < mano_landmarks.landmark[i - 2].y:
            dedos_arriba.append(True)
        else:
            dedos_arriba.append(False)

    if all(dedos_arriba):
        return "papel"
    elif not any(dedos_arriba):
        return "piedra"
    elif dedos_arriba[0] and dedos_arriba[1] and not dedos_arriba[2] and not dedos_arriba[3]:
        return "tijeras"
    else:
        return "desconocido"

@app.post("/detectar/")
async def detectar_gesto_en_imagen(file: UploadFile = File(...)):
    # Convertir archivo a imagen
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)

    gesto_detectado = "desconocido"
    
    if resultados.multi_hand_landmarks:
        for mano_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, mano_landmarks, mp_hands.HAND_CONNECTIONS)
            gesto_detectado = detectar_gesto(mano_landmarks)

    return {"gesto": gesto_detectado}

# Ejecutar la aplicación con uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Usa el puerto asignado por Render o el 10000 por defecto
    uvicorn.run(app, host="0.0.0.0", port=port)
