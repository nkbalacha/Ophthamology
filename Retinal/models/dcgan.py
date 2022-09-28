import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


device = 'cuda'
def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_c =3
        self.ngf = 32
        self.nz = 128
        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(self.nz, self.ngf*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.ngf*8, self.ngf*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.ngf*4, self.ngf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.ngf*2, self.ngf,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.ngf, self.ngf,
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.ngf)

        self.tconv6 = nn.ConvTranspose2d(self.ngf, self.n_c,
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.relu(self.bn5(self.tconv5(x)))

        x = F.tanh(self.tconv6(x))

        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_c =3
        self.ndf = 32
        # Input Dimension: (nc) x 128 x 128
        self.conv1 = nn.Conv2d(self.n_c, self.ndf,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 64 x 64
        self.conv2 = nn.Conv2d(self.ndf, self.ndf,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ndf)

        # Input Dimension: (ndf) x 32 x 32
        self.conv3 = nn.Conv2d(self.ndf, self.ndf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv4 = nn.Conv2d(self.ndf*2, self.ndf*4,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv5 = nn.Conv2d(self.ndf*4, self.ndf*8,
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.ndf*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv6 = nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)

        x = F.sigmoid(self.conv6(x))

        return x

criterion = nn.BCELoss()

def noise(n, n_features=128):
    return Variable(torch.randn(n, n_features)).to(device)

def make_ones_4d(size):
    data = Variable(torch.ones(size, 1,1,1))
    return data.to(device)

def make_zeros_4d(size):
    data = Variable(torch.zeros(size, 1,1,1))
    return data.to(device)

def train_discriminator(optimizer, real_data, fake_data,dis):
    n = real_data.size(0)

    optimizer.zero_grad()
    
    prediction_real = dis(real_data)

    error_real = criterion(prediction_real, make_ones_4d(n))
    error_real.backward()

    prediction_fake = dis(fake_data)
    error_fake = criterion(prediction_fake, make_zeros_4d(n))
    
    error_fake.backward()
    optimizer.step()
    
    return error_real + error_fake

def train_generator(optimizer, fake_data,dis):
    n = fake_data.size(0)
    optimizer.zero_grad()
    
    prediction = dis(fake_data)
    error = criterion(prediction, make_ones_4d(n))
    
    error.backward()
    optimizer.step()
    
    return error