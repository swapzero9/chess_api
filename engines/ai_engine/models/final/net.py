import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.start_l1 = nn.Linear(85, 128)
        self.start_l2 = nn.Linear(128, 256)
        self.start_n1 = nn.LayerNorm(256)

        self.start_ls1 = nn.LSTM(256, 256, 10)
        self.start_l3 = nn.Linear(256, 512)
        self.start_n2 = nn.LayerNorm(512)
        self.start_ls2 = nn.LSTM(512, 512, 10)
        self.start_t = nn.Tanh()

        self.drop = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1500)
        self.fc3 = nn.Linear(1500, 1968)

    def forward(self, x1, x2):
        x = torch.cat((
            torch.flatten(x1, 1),
            torch.flatten(x2, 1)
        ), dim=1)
        x = x.unsqueeze(1)
        
        x = self.start_n1(F.mish(self.start_l2(F.mish(self.start_l1(x)))))
        x, _ = self.start_ls1(x)
        x = self.start_n2(F.mish(self.start_l3(x)))
        x, _ = self.start_ls2(x)
        x = self.start_t(x)
        x = x.squeeze(1)
        
        x = F.mish(self.fc1(self.drop(x)))
        x = F.mish(self.fc2(self.drop(x)))
        x = F.softmax(self.fc3(x), dim=1)
        return x
