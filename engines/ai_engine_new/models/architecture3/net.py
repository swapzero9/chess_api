import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, transform):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=85, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.lstm1 = nn.LSTM(64, 64, 3)
        self.lstm2 = nn.LSTM(128, 128, 3)
        self.lstm3 = nn.LSTM(256, 256, 3)

        self.fc4 = nn.Linear(in_features=256, out_features=1024)

        self.action_head1 = nn.Linear(in_features=1024, out_features=1500)
        self.action_head2 = nn.Linear(in_features=1500, out_features=1968)
        self.value_head = nn.Linear(in_features=1024, out_features=1)

        self.transform = transform

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.transform(x)

        x = F.mish(self.fc1(x))
        x, _ = self.lstm1(x)
        x = F.mish(self.fc2(x))
        x, _ = self.lstm2(x)
        x = F.mish(self.fc3(x))
        x, _ = self.lstm3(x)
        x = F.mish(self.fc4(x))

        action_logits = self.action_head2(F.mish(self.action_head1(x)))
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)

    def predict(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = self.transform(x)

            x = F.mish(self.fc1(x))
            x.unsqueeze_(0)
            x, _ = self.lstm1(x)
            x.squeeze_(0)
            x = F.mish(self.fc2(x))
            x.unsqueeze_(0)
            x, _ = self.lstm2(x)
            x.squeeze_(0)
            x = F.mish(self.fc3(x))
            x.unsqueeze_(0)
            x, _ = self.lstm3(x)
            x.squeeze_(0)
            x = F.mish(self.fc4(x))

            action_logits = self.action_head2(F.mish(self.action_head1(x)))
            value_logit = self.value_head(x)

            return F.softmax(action_logits, dim=1), torch.tanh(value_logit)