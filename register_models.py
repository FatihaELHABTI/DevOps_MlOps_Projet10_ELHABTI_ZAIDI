import mlflow
import os

# Configuration
# On s'assure d'écrire dans le dossier mlruns à la racine
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Projet_10_Edge_Optimization")

# Chemins des modèles existants
model_int8 = "models/model_int8.tflite"
model_float = "models/model_float32.tflite"

print("🔄 Tentative d'enregistrement dans MLflow...")

if not os.path.exists(model_int8) or not os.path.exists(model_float):
    print("❌ ERREUR : Les fichiers modèles sont introuvables dans le dossier 'models/'")
    exit()

# On crée une NOUVELLE entrée propre
with mlflow.start_run(run_name="Livrable_Final_Models"):
    
    # 1. On logue les métriques (On remet les valeurs qu'on avait trouvées)
    print("📝 Log des métriques...")
    mlflow.log_metric("size_float32_mb", 11.68)
    mlflow.log_metric("size_int8_mb", 4.07)
    mlflow.log_metric("compression_ratio", 2.86)
    
    # 2. On logue les fichiers (Artifacts)
    print("📦 Upload du modèle Int8...")
    mlflow.log_artifact(model_int8, artifact_path="models_files")
    
    print("📦 Upload du modèle Float32...")
    mlflow.log_artifact(model_float, artifact_path="models_files")

print("✅ SUCCÈS ! Les modèles sont enregistrés.")
print("Rafraîchissez votre page MLflow maintenant.")