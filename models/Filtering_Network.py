import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Ada_fil2_pointnet(nn.Module):
    def __init__(self, channel):
        super(Ada_fil2_pointnet, self).__init__()

        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = torch.sum(x, 2, keepdim=True)
        x = x.view(-1, 512)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Ada_fil2_linear(nn.Module):
    def __init__(self, length):
        super(Ada_fil2_linear, self).__init__()
        self.fc1 = nn.Linear(length, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    a = torch.zeros(1, 2, 128)
    net = Ada_fil2_pointnet(2)
    print(net(a).shape)