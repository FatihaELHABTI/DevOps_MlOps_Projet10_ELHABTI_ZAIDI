import streamlit as st
import requests
import pandas as pd
import time
import plotly.express as px
import mlflow

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Edge Vision Security",
    page_icon="🔒",
    layout="wide"
)

# --- 1. GESTION DE LA SÉCURITÉ ---
USERS = {
    "admin": "admin123",
    "fatiha": "projet10"
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def check_login():
    user = st.session_state["username"]
    pwd = st.session_state["password"]
    if user in USERS and USERS[user] == pwd:
        st.session_state.authenticated = True
        del st.session_state["password"]
    else:
        st.error("Identifiants incorrects")

if not st.session_state.authenticated:
    st.title("🔒 Accès Restreint")
    st.markdown("Veuillez vous authentifier pour accéder au contrôle de la caméra.")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.text_input("Utilisateur", key="username")
        st.text_input("Mot de passe", type="password", key="password")
        st.button("Se connecter", on_click=check_login)
    st.stop()

# --- 2. LOGOUT ---
with st.sidebar:
    st.success(f"Connecté en tant que : **{st.session_state.username}**")
    if st.button("Déconnexion"):
        st.session_state.authenticated = False
        st.rerun()

# --- 3. DASHBOARD ---

API_URL = "http://localhost:8000"
MODELS = {
    "MODE RAPIDE (Int8)": "models/model_int8.tflite",
    "MODE PRÉCIS (Float32)": "models/model_float32.tflite"
}

def get_metrics():
    try:
        return requests.get(f"{API_URL}/metrics", timeout=1).json()
    except:
        return None

def send_ota(model_path):
    try:
        with st.spinner(f"Déploiement OTA en cours : {model_path}..."):
            res = requests.post(f"{API_URL}/update-model", params={"model_path": model_path})
            if res.status_code == 200:
                st.success("Mise à jour réussie !")
                try:
                    mlflow.set_tracking_uri("file:./mlruns") 
                    mlflow.set_experiment("Projet_10_Deployments")
                    with mlflow.start_run(run_name="OTA_Update"):
                        mlflow.log_param("user", st.session_state.username)
                        mlflow.log_param("model", model_path)
                except:
                    pass
                time.sleep(1)
            else:
                st.error("Erreur serveur")
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")

# Initialisation historiques
if "history_fps" not in st.session_state: st.session_state.history_fps = []
if "history_latency" not in st.session_state: st.session_state.history_latency = []
if "history_time" not in st.session_state: st.session_state.history_time = []

st.title("Edge Vision : Centre de Contrôle")
st.markdown("---")

col_control, col_monitor = st.columns([1, 2])

with col_control:
    st.subheader("Commandes OTA")
    st.info("Changez le modèle d'IA à distance.")
    selected_model = st.radio("Choisir le modèle à déployer :", list(MODELS.keys()))
    if st.button("LANCER LA MISE À JOUR", use_container_width=True):
        send_ota(MODELS[selected_model])

    st.markdown("---")
    st.subheader("État du Device")
    metric_container = st.empty()

with col_monitor:
    st.subheader("Performance Temps Réel")
    
    # --- MODIFICATION ICI : DEUX PLACEHOLDERS ---
    chart_fps = st.empty() # Emplacement pour le graphe FPS
    chart_lat = st.empty() # Emplacement pour le graphe Latence
    
    st.markdown(f"**Flux Vidéo Direct :** [Ouvrir le Stream]({API_URL}/video_feed)")
    st.image(f"{API_URL}/video_feed", caption="Live Edge View", width=400)

while True:
    metrics = get_metrics()
    if metrics:
        fps = metrics.get('fps', 0)
        lat = metrics.get('latency_ms', 0)
        version = metrics.get('model_version', 'Inconnu')
        obj = metrics.get('objects_detected', 0)

        with metric_container.container():
            st.metric("Modèle Actif", version)
            c1, c2 = st.columns(2)
            c1.metric("FPS", f"{fps:.2f}")
            c2.metric("Objets", obj)
            st.metric("Latence", f"{lat:.0f} ms")

        current_time = time.strftime("%H:%M:%S")
        st.session_state.history_time.append(current_time)
        st.session_state.history_fps.append(fps)
        st.session_state.history_latency.append(lat)

        if len(st.session_state.history_time) > 30:
            st.session_state.history_time.pop(0)
            st.session_state.history_fps.pop(0)
            st.session_state.history_latency.pop(0)

        # --- MODIFICATION ICI : DATAFRAME COMPLET ---
        df = pd.DataFrame({
            "Temps": st.session_state.history_time,
            "FPS": st.session_state.history_fps,
            "Latence": st.session_state.history_latency
        })

        # --- GRAPHIQUE 1 : FPS (Bleu) ---
        with chart_fps:
            fig_fps = px.line(df, x="Temps", y="FPS", title="Fluidité (FPS)", markers=True)
            fig_fps.update_layout(yaxis_range=[0, 20], height=250)
            st.plotly_chart(fig_fps, use_container_width=True)

        # --- GRAPHIQUE 2 : LATENCE (Rouge) ---
        with chart_lat:
            fig_lat = px.line(df, x="Temps", y="Latence", title="Latence Système (ms)", markers=True)
            fig_lat.update_traces(line_color='#FF4B4B') # Couleur rouge pour l'alerte
            fig_lat.update_layout(height=250)
            st.plotly_chart(fig_lat, use_container_width=True)

    else:
        metric_container.error("⚠️ Perte de signal...")
    
    time.sleep(1)