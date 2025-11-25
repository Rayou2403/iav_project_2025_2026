import numpy as np
import librosa

def shift_pitch(audio, sr, steps):
    factor = 2 ** (steps / 12)
   
    stretched = librosa.resample(audio, orig_sr=sr, target_sr=int(sr * factor))
    
    resampled = librosa.resample(stretched, orig_sr=int(sr * factor), target_sr=sr)
    return resampled

# Time-stretch stable : resample simple
def stretch_time(audio, sr, rate):
    new_len = int(len(audio) / rate)
    stretched = np.interp(
        np.linspace(0, len(audio), new_len),
        np.arange(len(audio)),
        audio
    )
    return stretched

# Bruit blanc
def add_noise(audio, level=0.005):
    noise = np.random.randn(len(audio))
    return audio + level * noise

def make_augmented_versions(audio, sr=16000):
    out = {}
    
    out["pitch_up"] = shift_pitch(audio, sr, 3)
    out["pitch_down"] = shift_pitch(audio, sr, -3)

    out["fast"] = stretch_time(audio, sr, 1.15)
    out["slow"] = stretch_time(audio, sr, 0.85)

    out["noisy"] = add_noise(audio)

    return out
