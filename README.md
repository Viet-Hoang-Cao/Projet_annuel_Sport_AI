Real-Time Fitness Form Analyzer
Posture Classification using Mediapipe + LSTM (PyTorch)

Ce projet a pour objectif de détecter automatiquement la bonne ou mauvaise exécution de mouvements sportifs en temps réel.
Il s’appuie sur Mediapipe Pose, un pipeline complet de preprocessing → séquences → entraînement, et un LSTM PyTorch pour classer la qualité d’un mouvement.

Il supporte aujourd’hui 4 exercices :

- Plank

- Push-up

- Squat

- Lunge

Les exercices possèdent des vidéos Good Form et Bad Form (pour l'instant, seulement les bad forms de plank).

**Objectifs du projet**

Classifier automatiquement la qualité d’un mouvement

Étendre la détection à 4 exercices

Comparer deux stratégies de features :

*Full-body :* 33 landmarks Mediapipe

*Lite-body :* seulement les points clés pertinents par exercice

Permettre une prédiction en temps réel via webcam

Avoir un pipeline robuste, réutilisable et modulaire

*Bad Forms pris en compte:*

- Plank

Hips trop hauts

Hips trop bas

Dos creusé

Alignement tête–tronc incorrect

- Push-up

Coudes trop ouverts

Dos non aligné

Amplitude incomplète

Hips qui tombent

- Pull-up

Kipping/balancement excessif

Amplitude insuffisante

Dos trop cambré

Menton ne dépasse pas la barre

- Lunge

Genou avant dépasse la pointe du pied

Genou arrière qui touche le sol

Inclinaison du torse

Pieds mal alignés

**Pipeline complète**
1. Extraction Mediapipe → CSV

Pour chaque vidéo :

Extraction 33 landmarks

Sauvegarde format :

x0, y0, z0, v0, x1, y1, z1, v1, … x32, y32, z32, v32, label


Puis extraction uniquement des points clés pertinents pour chaque exercice.

2. Création des séquences (sliding window)

Paramètre	Valeur
SEQ_LEN	30–50
STEP	5

Sortie :
exercise_sequences.npz contenant X et y.

3. Entraînement LSTM (PyTorch)

- Deux modèles :

Modèle	Description
lstm_full.pt	33 landmarks
lstm_selected.pt	seulement les points clés

Sorties :

modèle .pt

courbes d’entraînement

métriques par classe

erreurs détaillées

4. Détection en temps réel

python realtime/realtime_inference.py

Fonctionnalités :

Webcam → Mediapipe → Séquence → LSTM

Affichage direct GOOD / BAD

Affichage score de confiance

Option : changer le modèle (full / selected)

**Points clés sélectionnés (mode “lite-body”)**
- *Plank*

Épaules

Hanches

Genoux

Chevilles

- *Push-up*

Épaules

Coudes

Poignets

Hanches

- *Squat*

Épaules

Coudes

Poignets

- *Lunge*

Hanches

Genoux

Chevilles

**But :**
➡ réduire la dimension des features

➡ accélérer l’entraînement

➡ potentiellement augmenter la robustesse


Conseils pour prendre de bonnes vidéos

60 fps si possible

Plan complet du corps

Lumière stable

Plans variés (face / profil)

Répéter chaque BAD FORM clairement et de manière exagérée

Enregistrer plusieurs personnes pour plus de robustesse

➡ 10 à 20 vidéos GOOD par exercice

➡ 10 à 20 vidéos BAD par exercice
pour une base solide.

**INSTALLATIONS OBLIGATOIRES**
- Python 3.10+

- PyTorch
```
pip install torch torchvision torchaudio
```

- Pandas
Pour charger et nettoyer les CSV :

```
pip install pandas
```

- NumPy
Pour manipuler les séquences :

```
pip install numpy
```

- scikit-learn
Pour train/test split, normalisation (si besoin) :

```
pip install scikit-learn
```

- Jupyter Notebook / JupyterLab
Pour exécuter les notebooks :

```
pip install notebook jupyterlab
```

2. INSTALLATIONS OPTIONNELLES
Matplotlib

Pour tracer les courbes Loss/Accuracy :

```
pip install matplotlib
```

- OpenCV
Quand tu feras la prédiction en REALTIME :

```
pip install opencv-python
```

- TensorBoard
Visualisation d’apprentissage (même avec PyTorch) :

```
pip install tensorboard
```
3. INSTALLATIONS SPÉCIALES

- Mediapipe
Indispensable pour extraire les landmarks depuis les vidéos.

**ATTENTION** : Mediapipe n’est pas compatible avec toutes les versions de Python.
Compatible actuellement jusqu'à Python 3.11.

```
pip install mediapipe
```

3. Installer les dépendances principales
```
pip install mediapipe opencv-python pandas numpy torch torchvision torchaudio matplotlib tqdm jupyter notebook
```