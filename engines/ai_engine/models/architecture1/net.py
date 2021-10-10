import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 50, 2)
        self.conv2 = nn.Conv2d(50, 4000, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4000 + 21, 3600)
        self.fc2 = nn.Linear(3600, 1968)

    def forward(self, x1, x2):
        x1 = self.pool(F.mish(self.conv1(x1)))
        x1 = self.pool(F.mish(self.conv2(x1)))
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        x = F.mish(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x