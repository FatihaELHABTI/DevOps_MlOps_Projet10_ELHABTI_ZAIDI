# On part d'une version légère de Python (Slim)
FROM python:3.9-slim

# On définit le dossier de travail dans le conteneur
WORKDIR /app

# --- CORRECTION ICI ---
# Remplacement de libgl1-mesa-glx par libgl1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des librairies Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le code du projet
COPY app/ ./app
COPY models/ ./models
COPY data/ ./data

# La commande qui se lance au démarrage du conteneur
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]