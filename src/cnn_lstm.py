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

    def forward(self, x, lengths):
        # x: [batch, T, 40]

        x = x.transpose(1, 2)  # [batch, 40, T]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [batch, T, 64]

        out, _ = self.lstm(x)  # out: [batch, T, 64]

        # Récupération de la dernière vraie frame (pas du padding)
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_outputs.append(out[i, lengths[i]-1])

        out = torch.stack(last_outputs)  # [batch, 64]

        return self.fc(out)

