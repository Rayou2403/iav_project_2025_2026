import librosa
import numpy as np

def extract_mfcc(audio, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )
    return mfcc.T   # shape (time, n_mfcc)

def extract_melspec(audio, sr=16000, n_mels=128):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
