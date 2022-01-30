import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(85, 512, 15)
        self.drop = nn.Dropout(p=0.01)

        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1500)
        self.fc3 = nn.Linear(1500, 1968)

    def forward(self, x1, x2):
        # x1 => chessboard
        # x2 => castling and stuff
        # flatten so that dim b x 85

        x = torch.cat((
            torch.flatten(x1, 1),
            torch.flatten(x2, 1)
        ), dim=1)
        
        x = x.unsqueeze(1)
        
        x, _ = self.lstm(x) # 3dim
        x = torch.tanh(self.drop(x))
        x = torch.flatten(x, 1)
        
        x = self.norm(x)
        
        # linear
        x = F.mish(self.fc1(self.drop(x)))

        # linear
        x = F.mish(self.fc2(self.drop(x)))

        # linear
        x = F.softmax(self.fc3(x), dim=1)

        return x
