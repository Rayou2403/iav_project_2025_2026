import os
import numpy as np
# On va augmenter la rapidité avec les multiprocessings
from multiprocessing import Pool, cpu_count
from src.preprocessing import preprocess
from src.features import extract_mfcc, extract_melspec

SR = 16000

src = "data/augmented"
dst_mfcc = "data/processed/mfcc"
dst_melspec = "data/processed/melspec"

os.makedirs(dst_mfcc, exist_ok=True)
os.makedirs(dst_melspec, exist_ok=True)

# Traitement d'un fichier 
def process_file(args):
    audio_path, emotion = args

    audio = preprocess(audio_path, sr=SR)
    if audio is None or len(audio) == 0:
        return

    mfcc = extract_mfcc(audio, sr=SR)
    mel = extract_melspec(audio, sr=SR)

    # Dossiers de destination
    mfcc_dir = os.path.join(dst_mfcc, emotion)
    mel_dir = os.path.join(dst_melspec, emotion)
    os.makedirs(mfcc_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)

    base_name = os.path.basename(audio_path).replace(".wav", "")

    np.save(os.path.join(mfcc_dir, base_name + ".npy"), mfcc)
    np.save(os.path.join(mel_dir, base_name + ".npy"), mel)


# Litse de tâches
tasks = []

for emotion in os.listdir(src):
    emotion_path = os.path.join(src, emotion)
    if not os.path.isdir(emotion_path):
        continue

    for fname in os.listdir(emotion_path):
        if fname.endswith(".wav"):
            full_path = os.path.join(emotion_path, fname)
            tasks.append((full_path, emotion))

print(f"{len(tasks)} fichiers à traiter.")
print(f"Utilisation de {cpu_count()} cœurs CPU.")

# Multiproecssing
with Pool(processes=cpu_count()) as pool:
    pool.map(process_file, tasks)

    # Important pour fermer les workers, à ne pas oublier
    pool.close()
    pool.join()

print("Features extraites !")
