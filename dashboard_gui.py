import streamlit as st
import requests
import pandas as pd
import time
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Edge Vision Control Center",
    page_icon="",
    layout="wide"
)

API_URL = "http://localhost:8000"
MODELS = {
    " MODE RAPIDE (Int8)": "models/model_int8.tflite",
    " MODE PRÉCIS (Float32)": "models/model_float32.tflite"
}

# --- FONCTIONS ---
def get_metrics():
    try:
        return requests.get(f"{API_URL}/metrics", timeout=1).json()
    except:
        return None

def send_ota(model_path):
    try:
        with st.spinner(f" Déploiement OTA en cours : {model_path}..."):
            res = requests.post(f"{API_URL}/update-model", params={"model_path": model_path})
            if res.status_code == 200:
                st.success(" Mise à jour réussie !")
                time.sleep(1) # Laisser le temps de lire
            else:
                st.error(" Erreur serveur")
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")

# --- INITIALISATION DE L'ÉTAT (Mémoire pour les graphiques) ---
if "history_fps" not in st.session_state:
    st.session_state.history_fps = []
if "history_latency" not in st.session_state:
    st.session_state.history_latency = []
if "history_time" not in st.session_state:
    st.session_state.history_time = []

# --- INTERFACE GRAPHIQUE ---
st.title(" Edge Vision : Centre de Contrôle")
st.markdown("---")

# Division de l'écran en 2 colonnes
col_control, col_monitor = st.columns([1, 2])

with col_control:
    st.subheader(" Commandes OTA")
    st.info("Changez le modèle d'IA à distance sans éteindre la caméra.")
    
    selected_model = st.radio("Choisir le modèle à déployer :", list(MODELS.keys()))
    
    if st.button(" LANCER LA MISE À JOUR (OTA)", use_container_width=True):
        send_ota(MODELS[selected_model])

    st.markdown("---")
    st.subheader(" État du Device")
    
    # Placeholder pour les métriques textuelles
    metric_container = st.empty()

with col_monitor:
    st.subheader(" Performance Temps Réel")
    # Placeholder pour les graphiques
    chart_fps = st.empty()
    chart_lat = st.empty()

    # Optionnel : Afficher le lien vidéo
    st.markdown(f"**Flux Vidéo Direct :** [Ouvrir le Stream]({API_URL}/video_feed)")
    # On peut même essayer d'afficher l'image directement (rafraichissement lent)
    st.image(f"{API_URL}/video_feed", caption="Live Edge View", width=400)


# --- BOUCLE DE RAFRAÎCHISSEMENT (AUTO-REFRESH) ---
while True:
    metrics = get_metrics()
    
    if metrics:
        # 1. Mise à jour des chiffres
        fps = metrics.get('fps', 0)
        lat = metrics.get('latency_ms', 0)
        version = metrics.get('model_version', 'Inconnu')
        obj = metrics.get('objects_detected', 0)

        with metric_container.container():
            st.metric("Modèle Actif", version)
            c1, c2 = st.columns(2)
            c1.metric("FPS", f"{fps:.2f}")
            c2.metric("Objets", obj)
            st.metric("Latence Inférence", f"{lat:.0f} ms")

        # 2. Mise à jour de l'historique pour les graphiques
        current_time = time.strftime("%H:%M:%S")
        st.session_state.history_time.append(current_time)
        st.session_state.history_fps.append(fps)
        st.session_state.history_latency.append(lat)

        # On garde seulement les 30 derniers points pour que le graphique reste lisible
        if len(st.session_state.history_time) > 30:
            st.session_state.history_time.pop(0)
            st.session_state.history_fps.pop(0)
            st.session_state.history_latency.pop(0)

        # 3. Création des Dataframes pour Plotly
        df = pd.DataFrame({
            "Temps": st.session_state.history_time,
            "FPS": st.session_state.history_fps,
            "Latence (ms)": st.session_state.history_latency
        })

        # 4. Affichage des graphiques
        with chart_fps:
            fig_fps = px.line(df, x="Temps", y="FPS", title="Stabilité des FPS", markers=True)
            fig_fps.update_layout(yaxis_range=[0, 20]) # Echelle fixe
            st.plotly_chart(fig_fps, use_container_width=True)
        
        # Note : On affiche Latence ou pas, selon la place, ici on met juste FPS pour l'exemple
        # Vous pouvez décommenter ci-dessous pour avoir le 2eme graph
        with chart_lat:
            fig_lat = px.line(df, x="Temps", y="Latence (ms)", title="Latence Système", markers=True)
            st.plotly_chart(fig_lat, use_container_width=True)

    else:
        metric_container.error("⚠️ Connexion perdue avec le Device Docker...")
    
    # Pause de 1 seconde avant la prochaine mise à jour
    time.sleep(1)