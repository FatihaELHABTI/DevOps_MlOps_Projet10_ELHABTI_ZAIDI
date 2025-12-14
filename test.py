import cv2
from app.inference import EdgeDetector

# 1. Initialiser le détecteur avec le modèle INT8
detector = EdgeDetector("models/model_int8.tflite")

# 2. Charger une image de test (n'importe laquelle de calibration)
import os
image_files = os.listdir("data/calibration")
img_path = os.path.join("data/calibration", image_files[0])
img = cv2.imread(img_path)

# 3. Prédire
boxes, classes, scores, t_infer = detector.predict(img)

print(f"✅ Inférence réussie en {t_infer:.2f} ms")
print(f"Objets détectés (Score > 50%): {len([s for s in scores if s > 0.5])}")