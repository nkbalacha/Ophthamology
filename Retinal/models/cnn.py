import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_c =3
        self.ndf = 12
        # Input Dimension: (nc) x 128 x 128
        self.conv1 = nn.Conv2d(self.n_c, self.ndf,
            4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*4,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv5 = nn.Conv2d(self.ndf*4, self.ndf*4,
            4, 2, 1, bias=False)

        self.fc1 = nn.Linear(self.ndf*4*4*4,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)


    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = (F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

net = Net()