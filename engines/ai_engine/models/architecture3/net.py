import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(85, 250, 20)
        self.drop = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(250, 500)
        self.fc2 = nn.Linear(500, 750)
        self.fc3 = nn.Linear(750, 1968)

    def forward(self, x1, x2):
        # x1 => chessboard
        # x2 => castling and stuff
        # flatten so that dim b x 85

        x = torch.cat((
            torch.flatten(x1, 1),
            torch.flatten(x2, 1)
        ), dim=1)
        
        x.unsqueeze_(1)

        x, _ = self.lstm(x) # 3dim
        x = torch.flatten(x, 1)
        
        # linear
        x = F.mish(self.fc1(x))

        # linear
        x = F.mish(self.fc2(x))

        # linear
        x = F.log_softmax(self.fc3(x))

        return x
