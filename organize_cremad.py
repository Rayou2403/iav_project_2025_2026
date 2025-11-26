import os
import shutil

# Dossiers d'entrée / sortie
src = "data/raw/AudioWAV"
dst = "data/raw/organized"

# Dictionnaire pour traduire les codes émotion → nom clair
emotion_codes = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# Création du dossier organisé
os.makedirs(dst, exist_ok=True)

for fname in os.listdir(src):
    if not fname.endswith(".wav"):
        continue

    parts = fname.split("_")
    emotion_code = parts[2]  
    emotion = emotion_codes.get(emotion_code)

    if emotion is None:
        continue

    target_folder = os.path.join(dst, emotion)
    os.makedirs(target_folder, exist_ok=True)

    shutil.copy(os.path.join(src, fname), os.path.join(target_folder, fname))

print("OK : fichiers classés.")
