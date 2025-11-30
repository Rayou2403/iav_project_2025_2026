import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()

        # CNN sur les MFCC (40 coefficients)
        self.conv = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM pour la dynamique temporelle
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Classif finale
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x shape: (batch, time, 40)
        x = x.transpose(1, 2)      
        x = self.conv(x)
        x = x.transpose(1, 2)      

        out, _ = self.lstm(x)      
        out = out[:, -1]           

        return self.fc(out)
