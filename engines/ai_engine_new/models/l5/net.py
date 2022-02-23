import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, transform):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=85, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=512)
        self.lstm1 = nn.LSTM(128, 128, 3)
        self.lstm2 = nn.LSTM(256, 256, 3)
        self.lstm3 = nn.LSTM(512, 512, 3)
        self.drop = nn.Dropout(0.05)
        self.prob_seq = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 1024), nn.Mish(),
            nn.LayerNorm(1024), nn.Linear(1024, 2048), nn.Mish(),
            nn.LayerNorm(2048), nn.Linear(2048, 1968)
        )
        self.value_seq1 = nn.Sequential(
            nn.Linear(85, 128), nn.Mish(), nn.LayerNorm(128),
            nn.Linear(128, 256), nn.Mish(), nn.LayerNorm(256),
        )
        self.value_lstm = nn.LSTM(256, 256, 5, bidirectional=True)
        self.value_seq2 = nn.Sequential(
            nn.Linear(512, 128), nn.Mish(), nn.LayerNorm(128),
            nn.Linear(128, 1), nn.Tanh(),
        )

        self.transform = transform

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.transform(x)

        x1 = F.mish(self.fc1(x))
        x1, _ = self.lstm1(x1)
        x1 = F.mish(self.fc2(self.drop(x1)))
        x1, _ = self.lstm2(x1)
        x1 = F.mish(self.fc3(self.drop(x1)))
        x1, _ = self.lstm3(x1)
        x1.squeeze_(0)
        x1 = self.prob_seq(x1)

        x2 = self.value_seq1(x)
        x2, _ = self.value_lstm(x2)
        x2 = self.value_seq2(x2)

        return F.softmax(x1, dim=1), x2

    def predict(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = self.transform(x)

            x.unsqueeze_(0)
            x1 = F.mish(self.fc1(x))
            x1, _ = self.lstm1(x1)
            x1 = F.mish(self.fc2(self.drop(x1)))
            x1, _ = self.lstm2(x1)
            x1 = F.mish(self.fc3(self.drop(x1)))
            x1, _ = self.lstm3(x1)
            x1.squeeze_(0)
            x1 = self.prob_seq(x1)

            x2 = self.value_seq1(x)
            x2, _ = self.value_lstm(x2)
            x2 = self.value_seq2(x2)

            return F.softmax(x1, dim=1), x2
