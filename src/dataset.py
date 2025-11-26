import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F   

EMOTION_TO_IDX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

class SpeechDataset(Dataset):
    def __init__(self, feature_dir, feature_type="mfcc"):
        self.feature_dir = feature_dir
        self.feature_type = feature_type
        self.samples = []

        for emotion in os.listdir(feature_dir):
            emo_path = os.path.join(feature_dir, emotion)
            if not os.path.isdir(emo_path):
                continue

            label = EMOTION_TO_IDX.get(emotion)
            if label is None:
                continue

            for fname in os.listdir(emo_path):
                if fname.endswith(".npy"):
                    full_path = os.path.join(emo_path, fname)
                    self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        x = np.load(feature_path)
        x = torch.tensor(x, dtype=torch.float32)

        if self.feature_type == "melspec":
           
            target_len = 128
            T = x.shape[1]

            if T < target_len:
                x = F.pad(x, (0, target_len - T))
            else:
                x = x[:, :target_len]

            x = x.unsqueeze(0)

        return x, torch.tensor(label, dtype=torch.long)
