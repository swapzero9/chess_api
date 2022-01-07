import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 30, 1) 
        self.conv2 = nn.Conv2d(30, 40, 2) 
        self.conv3 = nn.Conv2d(40, 50, 2) # 50 x 

        self.drop = nn.Dropout(p=0)
        self.batch_norm1 = nn.BatchNorm2d(30) # maybe someday
        self.batch_norm2 = nn.BatchNorm2d(40) # maybe someday
        self.batch_norm3 = nn.BatchNorm2d(50) # maybe someday

        self.fc1 = nn.Linear(1821, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, xx):
        xx[0] = F.mish(self.conv1(xx[0]))
        xx[0] = self.batch_norm1(xx[0])
        xx[0] = F.mish(self.conv2(xx[0]))
        xx[0] = self.batch_norm2(xx[0])
        xx[0] = F.mish(self.conv3(xx[0]))
        xx[0] = self.batch_norm3(xx[0])

        x = torch.cat((
            torch.flatten(xx[0], 1), 
            torch.flatten(xx[1], 1)
        ), dim=1)
        x = F.relu(self.fc1(self.drop(x)))
        x = F.relu(self.fc2(self.drop(x)))
        x = F.relu(self.fc3(self.drop(x)))
        return x