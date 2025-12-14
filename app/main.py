from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import threading
import time
import os
import io
from app.inference import EdgeDetector

app = FastAPI(title="Edge Vision System")

# --- Config ---
# On commence par défaut avec le modèle rapide
CURRENT_MODEL = "models/model_int8.tflite"
VIDEO_SOURCE = "data/video_test.mp4"

# --- Variables Globales ---
detector = None
stop_thread = False
latest_frame_processed = None 
lock = threading.Lock() 

telemetry = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "model_version": "v1_int8",
    "objects_detected": 0
}

def video_processing_loop():
    """Boucle principale : Lit vidéo -> Détecte -> Dessine"""
    global detector, telemetry, stop_thread, latest_frame_processed
    
    print(f"🎥 Démarrage du flux vidéo : {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if detector:
            try:
                # 1. Inférence
                boxes, classes, scores, latency = detector.predict(frame)
                
                # 2. Dessin (Filtre 15%)
                count = 0
                h, w, _ = frame.shape
                for i in range(len(scores)):
                    if scores[i] > 0.15:  
                        count += 1
                        ymin, xmin, ymax, xmax = boxes[i]
                        start = (int(xmin * w), int(ymin * h))
                        end = (int(xmax * w), int(ymax * h))
                        
                        cv2.rectangle(frame, start, end, (0, 255, 0), 2)
                        label = f"ID {int(classes[i])}: {scores[i]:.2f}"
                        cv2.putText(frame, label, (start[0], start[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 3. Mise à jour Télémétrie
                telemetry["latency_ms"] = latency
                telemetry["objects_detected"] = count
                if latency > 0:
                    telemetry["fps"] = 1000.0 / latency
                
                # 4. Sauvegarder pour le web
                with lock:
                    latest_frame_processed = frame.copy()
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Erreur boucle : {e}")
                time.sleep(0.1)

    cap.release()
    print("🛑 Arrêt du flux vidéo (Thread terminé).")

def generate_mjpeg():
    """Générateur web"""
    global latest_frame_processed
    while True:
        with lock:
            if latest_frame_processed is None:
                time.sleep(0.1)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", latest_frame_processed)
            if not flag: continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        time.sleep(0.05)

@app.on_event("startup")
def startup_event():
    global detector
    if os.path.exists(CURRENT_MODEL):
        detector = EdgeDetector(CURRENT_MODEL)
        print("✅ Modèle chargé.")
        # On lance le thread vidéo
        thread = threading.Thread(target=video_processing_loop, daemon=True)
        thread.start()
    else:
        print(f"❌ Modèle introuvable : {CURRENT_MODEL}")

@app.get("/")
def index():
    return "Système Prêt. Routes: /video_feed, /metrics, /update-model"

@app.get("/metrics")
def get_metrics():
    return telemetry

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_mjpeg(), 
                             media_type="multipart/x-mixed-replace;boundary=frame")

# --- LA PARTIE CORRIGÉE POUR L'OTA ---
@app.post("/update-model")
def update_model(model_path: str):
    global detector, stop_thread, telemetry
    
    print(f"📥 Demande OTA reçue pour : {model_path}")
    
    if not os.path.exists(model_path):
        return {"status": "error", "message": "Fichier introuvable"}

    try:
        # 1. On signale au thread vidéo de s'arrêter
        stop_thread = True
        
        # 2. On attend un peu qu'il s'arrête proprement
        time.sleep(1) 
        
        # 3. On charge le nouveau modèle
        print("🔄 Rechargement du moteur IA...")
        detector = EdgeDetector(model_path)
        
        # On met à jour le nom dans le JSON pour le dashboard
        telemetry["model_version"] = os.path.basename(model_path)
        
        # 4. On REDÉMARRE un nouveau thread vidéo (car l'ancien est mort)
        stop_thread = False
        new_thread = threading.Thread(target=video_processing_loop, daemon=True)
        new_thread.start()
        
        print("✅ OTA Terminée avec succès.")
        return {"status": "success", "message": f"Modèle basculé sur {model_path}"}
        
    except Exception as e:
        print(f"❌ Erreur OTA : {e}")
        return {"status": "error", "message": str(e)}