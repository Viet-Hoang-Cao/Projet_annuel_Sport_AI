# Project — Plank Form Detection Using Mediapipe + MLP (Full & Keypoints)

## Introduction
Ce projet a pour objectif de détecter automatiquement si un exercice de **plank** (gainage) est réalisé en **bonne** ou **mauvaise** forme grâce à l’IA.  
Pour cela, nous utilisons :

- **Mediapipe** pour extraire les points clés (landmarks) du corps humain
- **MLP (Multi-Layer Perceptron)** pour classer les postures
- Deux approches :  
  - **FULL landmarks (33 points × 4 valeurs)**  
  - **KEYPOINTS (17 points essentiels × 4 valeurs)**  

Le pipeline se compose de plusieurs notebooks, chacun ayant un rôle précis.

---

# Architecture générale

project_annuel/
│
├── data/
│ ├── datasets/
│ │ ├── plank/
│ │ │ ├── good/ # Vidéos bonne forme
│ │ │ ├── bad/ # Vidéos mauvaise forme
│
├── core/
│ ├── plank_model/
│ │ ├── data/
│ │ │ ├── plank_dataset_full.csv
│ │ │ ├── plank_dataset_keypoints.csv
│ │ │ ├── scaler_full.pkl
│ │ │ ├── scaler_keypoints.pkl
│ │ │ ├── plank_test.csv
│ │ ├── model/
│ │ │ ├── plank_mlp_full.pt
│ │ │ ├── plank_mlp_keypoints.pt
│ │ ├── notebooks/
│ │ │ ├── data_plank.ipynb
│ │ │ ├── scaler_and_test_plank.ipynb
│ │ │ ├── mlp_train_full.ipynb
│ │ │ ├── mlp_train_keypoints.ipynb
│ │ │ ├── realtime_plank_mlp.ipynb
│
└── README.md

---

# Notebooks détaillés

## `data_plank.ipynb` — Extraction des landmarks

### Objectif
Convertir les vidéos (Good / Bad Form) en fichiers CSV exploitables pour l’entraînement des modèles.

### Librairies utilisées
| Librairie | Rôle |
|----------|------|
| `mediapipe` | Détection des 33 points du squelette humain |
| `opencv (cv2)` | Lecture vidéo frame par frame |
| `pandas` | Construction des datasets CSV |
| `numpy` | Manipulation numérique |
| `glob` | Parcours automatique des dossiers |
| `os` | Création des dossiers |

### Fonctionnement
Pour chaque vidéo :

1. Lecture des frames
2. Passage dans Mediapipe → extraction des landmarks
3. Extraction de :
   - **FULL (33 points)** → 132 valeurs (x, y, z, visibility)
   - **KEYPOINTS (17 points)** → 68 valeurs
4. Ajout du `label` (0 = good, 1 = bad)
5. Sauvegarde dans :
   - `plank_dataset_full.csv`
   - `plank_dataset_keypoints.csv`

### Pourquoi FULL + KEYPOINTS ?
- FULL = plus de précision, mais plus fragile (bruit, occlusions)
- KEYPOINTS = robustesse + performance MLP optimisée  
→ L’auteur initial de GitHub utilise aussi des keypoints essentiels.

---

## `scaler_and_test_plank.ipynb` — Standardisation + Dataset Test

### Objectif
Préparer les données pour l’entraînement du modèle.

### Rôle du scaler
Les valeurs des landmarks ont des échelles différentes :

- x ≈ 0.3
- y ≈ 0.7
- z ≈ -0.03
- visibilité ≈ 0.9

➡ Sans normalisation, le modèle favoriserait les colonnes les plus grandes.

### Actions réalisées
1. Chargement des deux datasets
2. Séparation X / y
3. Application de `StandardScaler()` :
   - moyenne = 0
   - variance = 1
4. Sauvegarde :
   - `scaler_full.pkl`
   - `scaler_keypoints.pkl`
5. Génération d’un fichier test :
   - `plank_test.csv`

### Pourquoi cette étape est essentielle ?
Le scaler doit être **réutilisé** pendant :
- l’entraînement
- la détection réelle
- l’inférence sur vidéo  

Sans scaler → le modèle donnerait des prédictions fausses.

---

## `mlp_train_full.ipynb` — Modèle MLP Full Landmarks  
## `mlp_train_keypoints.ipynb` — Modèle MLP Keypoints

### Objectif
Entraîner deux modèles différents :

- un avec **132 features**  
- un avec **68 features**

### Pourquoi un MLP ?
Un MLP est adapté quand :

- Les données sont tabulaires (CSV)
- L’ordre temporel n’est pas essentiel  
  (un plank est statique → pas besoin de LSTM)
- Le dataset est petit
- On veut un modèle rapide et léger

Un MLP est :
- simple
- efficace
- rapide à entraîner
- excellent pour la classification binaire posture

### 🛠 Actions du notebook
1. Chargement du dataset
2. Application du scaler
3. Split train/test
4. Définition du modèle MLP PyTorch
5. Entraînement :
   - forward pass
   - backward pass
   - optimisation Adam
6. Calcul de l’accuracy
7. Sauvegarde :
   - `plank_mlp_full.pt`
   - `plank_mlp_keypoints.pt`

### Notes importantes
- Le modèle keypoints est souvent plus stable.
- Le FULL peut sur-apprendre sur un petit dataset.

---

## `realtime_plank_mlp.ipynb` — Détection Vidéo / Webcam

### Objectif
Tester le modèle en conditions réelles.

### Fonctionnement
1. Chargement :
   - modèle MLP
   - scaler correspondant
2. Ouverture :
   - d’une vidéo (`cv2.VideoCapture(path)`)
   - ou de la webcam (`cv2.VideoCapture(0)`)
3. Extraction des keypoints en temps réel via Mediapipe
4. Transformation avec le scaler
5. Passage dans le modèle MLP → prédiction
6. Affichage OpenCV :
   - classe : **GOOD** ou **BAD**
   - probabilité
   - skeleton annoté

### À modifier si tu veux tester une vidéo
```python
cap = cv2.VideoCapture("chemin/vers/la/video.mp4")

Points importants

Mediapipe peut manquer des frames → normal

Le scaler doit être appliqué AVANT le modèle

La qualité vidéo influence énormément la détection