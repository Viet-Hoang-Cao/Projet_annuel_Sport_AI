import torch
import torch.nn as nn

class ExerciseLSTM(nn.Module):


    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=9, dropout=0.3):

        super(ExerciseLSTM, self).__init__()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Couche fully-connected pour sortir les classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # On prend la dernière frame du LSTM
        last_output = lstm_out[:, -1, :]

        # On génère logits
        logits = self.fc(last_output)

        return logits
