import requests
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Configuration
API_URL = "http://localhost:8000"
MODELS = {
    "RAPIDE": "models/model_int8.tflite",
    "LENT": "models/model_float32.tflite"
}

def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def send_update(model_path):
    print(f"\n🚀 ENVOI DE LA MISE À JOUR OTA VERS : {model_path} ...")
    try:
        # On envoie le chemin du modèle à l'API
        # Note: 'url' est le nom du paramètre dans main.py
        response = requests.post(f"{API_URL}/update-model", params={"model_path": model_path})
        print(f"Réponse du device : {response.json()}")
    except Exception as e:
        print(f"Erreur OTA : {e}")

def main():
    print("--- DÉMARRAGE DU DASHBOARD DE CONTRÔLE ---")
    print("1. Surveillance des performances")
    print("2. Changement de modèle à chaud")
    print("------------------------------------------")

    # Historique pour le graphique (optionnel, ici on fait du texte)
    try:
        while True:
            metrics = get_metrics()
            
            if metrics:
                fps = metrics.get('fps', 0)
                lat = metrics.get('latency_ms', 0)
                ver = metrics.get('model_version', 'Inconnu')
                obj = metrics.get('objects_detected', 0)
                
                # Affichage style "Tableau de bord"
                print(f"\r[DEVICE STATUS] Modèle: {ver} | FPS: {fps:.2f} | Latence: {lat:.0f}ms | Objets: {obj}   ", end="")
            
            # Simulation d'interaction
            # Si on veut changer de modèle manuellement, on pourrait ajouter un input ici
            # Mais pour l'instant on monitore juste.
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Interruption...")
        # Menu interactif à la fermeture
        choix = input("Voulez-vous changer le modèle avant de quitter ? (1: Rapide, 2: Lent, Enter: Non) : ")
        if choix == "1":
            send_update(MODELS["RAPIDE"])
        elif choix == "2":
            send_update(MODELS["LENT"])
        print("Fin du dashboard.")

if __name__ == "__main__":
    main()