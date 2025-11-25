import os
import numpy as np
from scipy.io.wavfile import write
from src.preprocessing import preprocess
from src.augment import make_augmented_versions

SR = 16000

src = "data/raw/organized"
dst = "data/augmented"

os.makedirs(dst, exist_ok=True)

for emotion in os.listdir(src):
    emotion_path = os.path.join(src, emotion)
    if not os.path.isdir(emotion_path):
        continue
    
    target_dir = os.path.join(dst, emotion)
    os.makedirs(target_dir, exist_ok=True)

    for fname in os.listdir(emotion_path):
        if not fname.endswith(".wav"):
            continue

        in_path = os.path.join(emotion_path, fname)

        audio = preprocess(in_path, sr=SR)

        # Ignore fichiers vides ou NaN
        if audio is None or len(audio) == 0:
            continue
        if not np.all(np.isfinite(audio)):
            continue

        # Enregistrer version "clean"
        clean_path = os.path.join(target_dir, fname)
        write(clean_path, SR, audio.astype("float32"))

        # Générer les augmentations
        aug_versions = make_augmented_versions(audio, sr=SR)

        for aug_name, aug_audio in aug_versions.items():
            if aug_audio is None or len(aug_audio) == 0:
                continue
            if not np.all(np.isfinite(aug_audio)):
                continue

            new_name = fname.replace(".wav", f"_{aug_name}.wav")
            out_path = os.path.join(target_dir, new_name)
            write(out_path, SR, aug_audio.astype("float32"))

print("Dataset augmenté généré.")
