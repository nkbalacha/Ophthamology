# -*- coding: utf-8 -*-
"""Vanilla_GAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WrqvBk4xFXOYehxXMCTCzAMT1X6xHx7F
"""

import sys
import math
import toml
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,Dataset
from torchvision.io import read_image
from torchvision import transforms as T, utils
from models.dcgan import Generator, Discriminator,noise,train_generator,train_discriminator, weights_init
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def main():
    batch_size = 512
    transform = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Grayscale(),
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ])
    to_image = transforms.ToPILImage()
    class Kaggle_Dataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 2])+".jpeg")
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
    #trainset = (_KAGGLE_LABEL_DIR,_KAGGLE_DATA_DIR, transform=transform)
    trainset = Kaggle_Dataset(annotations_file = _KAGGLE_LABEL_DIR,img_dir =_KAGGLE_DATA_DIR, transform = transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    generator = Generator()
    discriminator = Discriminator()
    device = 'cuda'
    generator.to(device)
    discriminator.to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    g_optim = optim.Adam(generator.parameters(), lr=2e-4)
    d_optim = optim.Adam(discriminator.parameters(), lr=2e-4)
    g_losses = []
    d_losses = []
    images = []

    criterion = nn.BCELoss()

    num_epochs = 250
    k = 1
    #test_noise = noise(64)
    nz=128
    test_noise = torch.randn(64, nz, 1, 1, device=device)

    #generator.train()
    #discriminator.train()
    for epoch in range(num_epochs):
        g_error = 0.0
        d_error = 0.0
        for i, data in enumerate(trainloader):
            imgs, _ = data
            n = len(imgs)
            for j in range(k):
                train_noise = torch.randn(n, nz, 1, 1, device=device)
                fake_data = generator(train_noise).detach()
                real_data = imgs.to(device)
                d_error += train_discriminator(d_optim, real_data, fake_data,discriminator)
            fake_noise = torch.randn(n, nz, 1, 1, device=device)
            fake_data = generator(fake_noise)
            g_error += train_generator(g_optim, fake_data,discriminator)

        img = generator(test_noise).cpu().detach()       
        #img = (.5*generator(test_noise)+.5).cpu().detach()
        img = make_grid(img)
        utils.save_image(img, _OUTPUT_DIR+str('_03.jpeg'), nrow = int(math.sqrt(64)))
        images.append(img)
        g_losses.append(g_error.cpu().detach().numpy()/i)
        d_losses.append(d_error.cpu().detach().numpy()/i)       
        imgs = [np.array(to_image(i).convert("RGB")) for i in images]
        imageio.mimsave(_OUTPUT_DIR+'progress_kaggle_dcgan.gif', imgs)
        torch.save(generator.state_dict(), _OUTPUT_DIR+'kaggle_dc_generator.pth')
        print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f}\r'.format(epoch, g_error/i, d_error/i))
    print('Training Finished')
    torch.save(generator.state_dict(), _OUTPUT_DIR+'kaggle_dc_generator.pth')

    #imgs = [np.array(to_image(i)) for i in images]
    imgs = [np.array(to_image(i).convert("RGB")) for i in images]

    imageio.mimsave(_OUTPUT_DIR+'progress_kaggle_dcgan.gif', imgs)

    plt.plot(g_losses, label='Generator_Losses')
    plt.plot(d_losses, label='Discriminator Losses')
    plt.legend()
    plt.savefig(_OUTPUT_DIR+'kaggle_dcgan_loss.png')


if __name__ == '__main__':

    # run command: time /home/user/miniconda/bin/python _01.py
	# takes ~40-45 minutes

	# create global variables from .toml config file
	cfg = toml.load('./config.toml')
	for section in cfg.keys():
		for key in cfg[section].keys():
			globals()[key] = cfg.get(section).get(key)

	# setup logging
	logging.basicConfig(
		filename=_INDEX_LOGFILE, 
		filemode='w', 
		format='%(message)s'
	)

	main()




    






