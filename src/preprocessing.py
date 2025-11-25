import librosa
import numpy as np

def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def normalize(audio):
    if len(audio) == 0:
        return audio
    m = np.max(np.abs(audio))
    if m == 0 or np.isnan(m):
        return audio
    return audio / m


def remove_silence(audio, sr=16000, top_db=25):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    cleaned = np.concatenate([audio[s:e] for (s, e) in intervals])
    return cleaned

def preprocess(path, sr=16000):
    audio = load_audio(path, sr)
    audio = remove_silence(audio, sr)

    if len(audio) == 0:
        return None

    audio = normalize(audio)

    if len(audio) == 0:
        return None

    return audio
