# Déploiement de Modèles d'Intelligence Artificielle sur Dispositifs Edge pour la Détection Automatique d'Intrusions

<img width="1032" height="592" alt="image" src="https://github.com/user-attachments/assets/1850da43-15d1-4861-935b-43153ecbcd41" />

## Aperçu du Projet
Ce projet porte sur le déploiement de modèles d'IA optimisés sur des dispositifs edge à ressources limitées pour la détection en temps réel d'intrusions humaines dans des applications de sécurité. Il illustre une approche **AIoT** (Artificial Intelligence of Things) en optimisant le modèle **SSD MobileNet V2** via une quantification post-entraînement (Int8) afin de réduire la taille du modèle et la latence d'inférence.

<img width="842" height="399" alt="image" src="https://github.com/user-attachments/assets/73e053d2-591e-4bdf-9937-dab05e7d2453" />

Le système simule un environnement Raspberry Pi à l'aide de conteneurs Docker avec limitations CPU. Il intègre une API FastAPI pour l'inférence, un dashboard Streamlit pour le monitoring et les mises à jour OTA, ainsi que MLflow pour le suivi du cycle de vie des modèles.

Les résultats valident qu'un modèle quantifié (4 Mo) est 2 à 3 fois plus rapide (11 FPS) que la version originale (11 Mo, 4 FPS) en environnement contraint, tout en maintenant une précision acceptable.

**Mots-clés :** Edge Computing, Quantification Int8, SSD MobileNet V2, TensorFlow Lite, FastAPI, Streamlit, MLflow, OTA, Docker, AIoT, MLOps.

<img width="870" height="486" alt="image" src="https://github.com/user-attachments/assets/a294229a-964d-49d8-b17b-0bd4daa8a4be" />


## Fonctionnalités Principales
- **Optimisation du Modèle** — Quantification post-entraînement de Float32 à Int8, réduisant la taille de ~65% et augmentant la vitesse de 140% sur edge.
- **Simulation Edge** — Conteneurs Docker imitant les contraintes d'un Raspberry Pi (0.5 cœur CPU, 512 Mo RAM).
- **Inférence en Temps Réel** — Backend FastAPI traitant les flux vidéo, détectant les intrusions avec TFLite et renvoyant un stream MJPEG annoté avec métadonnées JSON.
- **Dashboard Sécurisé** — Interface Streamlit avec authentification pour monitorer FPS/latence, visualiser le flux traité et déclencher des mises à jour OTA.
- **Mises à Jour OTA** — Switching sécurisé entre modèles (Float32 ↔ Int8) en moins de 2 secondes sans interruption de service.
- **Suivi MLOps** — MLflow pour logger les expériences, registre de modèles et audits de déploiement.
- **Pipeline CI/CD** — GitHub Actions pour tests automatisés, benchmarks sur x86/ARM64 et validation (ex. >10 FPS, <100ms latence).

## Technologies Utilisées
- **IA/ML** : TensorFlow Lite, OpenCV, SSD MobileNet V2 FPNLite 320x320 (pré-entraîné sur COCO 2017, calibré sur Penn-Fudan).
- **Backend** : FastAPI (inférence asynchrone et endpoints API).
- **Frontend** : Streamlit (dashboard interactif avec graphs et vidéo en temps réel).
- **MLOps** : MLflow (tracking, registry), DVC (versioning des modèles).
- **Containerisation** : Docker & Docker Compose.
- **CI/CD** : GitHub Actions.


## Architecture du Modèle
Le modèle choisi est **SSD MobileNet V2**, optimisé pour les environnements mobiles grâce aux convolutions séparables en profondeur.

<img width="1285" height="734" alt="image" src="https://github.com/user-attachments/assets/deb2e7c9-00ec-4f09-b9b5-66adfa691c76" />

## Pipeline du Projet

<img width="1215" height="807" alt="image" src="https://github.com/user-attachments/assets/e8c8f981-019f-405e-a48c-8751def3c5e2" />

une pipeline CI/CD complète et automatisée pour le déploiement de modèles TensorFlow Lite optimisés sur dispositifs Edge (comme des Raspberry Pi). Elle illustre un workflow professionnel MLOps avec tests multi-plateformes, validation de performance et déploiement conditionnel en production via OTA.
Voici une explication étape par étape :
1. Déclenchement du workflow
Le processus commence par une activité Git (typiquement GitHub) :

Un développeur pousse une branche feature/add ou ouvre une Pull Request.
Un Manual Dispatch est également possible (bouton manuel dans GitHub Actions).
Temps total estimé : ~12 minutes.

2. Étape 1 : Data & Environment Setup (Configuration initiale)
Cette étape commune prépare l’environnement :

Checkout du code
Installation de Python 3.11
Installation de DVC (Data Version Control)
Téléchargement des modèles depuis le stockage DVC
Upload des artefacts sous le dossier edge-models

3. Étapes parallèles de benchmarking (Tests multi-plateformes)
La pipeline teste le modèle sur trois environnements différents en parallèle :

2. x86 Cloud - FP32
Environnement puissant (Ubuntu latest)
Modèle non quantifié (précision flottante 32 bits)
Résultat : 206 ms par image → 4.8 FPS
3. x86 Cloud - INT8
Même machine puissante, mais modèle quantifié en Int8
Résultat : 233 ms → 4.3 FPS
(Légère dégradation due à la quantification, mais négligeable sur CPU puissant)
4. ARM64 Edge - INT8 (le plus important !)
Simulation réaliste d’un Raspberry Pi (image ubuntu-24.04-arm)
Modèle quantifié Int8
Résultat : 79 ms → 12.6 FPS 
Gain énorme par rapport au modèle FP32 sur edge

Toutes ces étapes uploadent leurs résultats au format JSON pour comparaison.
4. Étape 5 : Real Performance Dashboard (Validation et décision automatisée)
Cette étape centrale collecte tous les résultats et prend une décision de déploiement :

Télécharge tous les artefacts JSON
Crée un DataFrame Pandas
Génère un graphique Matplotlib
Produit un rapport Markdown

Règles de validation automatisées (gates de qualité) :

FPS > 10 sur edge ? 
Latence < 100 ms sur edge ? 
Si une condition échoue → Block Deployment 
Si toutes les conditions passent → Deploy to Production 

5. Déploiement final
En cas de succès :

Intégration avec MLflow Tracking pour logger les métriques, paramètres et artefacts

<img width="850" height="403" alt="image" src="https://github.com/user-attachments/assets/65174501-cc28-4231-8967-0e4eb8fb7490" />

Déploiement du modèle validé sur la flotte de production
Mise à jour Over-The-Air (OTA) sur 50 Raspberry Pi simultanément

Légende en bas : statut des jobs (en cours, succès, échec, en attente)
Comparaison avec YOLO :

<img width="1292" height="793" alt="image" src="https://github.com/user-attachments/assets/86fc7ff2-4196-4359-a324-42bbf36ca89a" />


## Avantages Edge vs Cloud

<img width="688" height="660" alt="image" src="https://github.com/user-attachments/assets/19fb2b54-8aa1-45e8-97eb-fdb3069b0ef5" />


## Processus de Quantification


<img width="721" height="863" alt="image" src="https://github.com/user-attachments/assets/d1be1a55-0d27-46b6-a43d-799a86c13264" />


## Résultats
| Modèle              | Taille | FPS (contraint) | Précision     |
|---------------------|--------|-----------------|---------------|
| Original (Float32)  | 11 Mo  | 4 FPS           | Haute         |
| Quantifié (Int8)    | 4 Mo   | 11 FPS          | Acceptable    |


## Installation et Exécution
1. Cloner le repository : `git clone https://github.com/votre-repo/projet10-edge-deployment.git`
2. Installer les dépendances : `pip install -r requirements.txt`
3. Lancer MLflow : `mlflow ui --port 5000`
4. Démarrer l'API : `uvicorn app.api:main --reload`
5. Lancer le dashboard : `streamlit run app/dashboard.py`
6. Simuler l'edge : `docker-compose up --build`

**Prérequis :** Python 3.8+, Docker, webcam ou vidéo test.

## Auteurs
- ZAIDI Malak
- ELHABTI Fatiha

**Encadrant :** Pr. HAMIDA Soufiane

Ce projet est réalisé dans le cadre du Master SDIA – Module MLOps & DevOps (Année 2025-2026). Contributions bienvenues via pull requests !
