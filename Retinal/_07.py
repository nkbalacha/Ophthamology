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
from models.cnn import Net
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def main():
    batch_size = 128
    transform = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Grayscale(),
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ])
    to_image = transforms.ToPILImage()
    class Long_Dataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]))
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
    #trainset = (_KAGGLE_LABEL_DIR,_KAGGLE_DATA_DIR, transform=transform)
    #trainset = Long_Dataset(annotations_file = _LONG_TRAIN_LABEL_DIR,img_dir =_LONG_TRAIN_DATA_DIR, transform = transform)
    #trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    #testset = Long_Dataset(annotations_file = _LONG_TEST_LABEL_DIR,img_dir =_LONG_TEST_DATA_DIR, transform = transform)
    #testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    dataset = Long_Dataset(annotations_file = _LONG_LABEL_DIR,img_dir =_LONG_DATA_DIR, transform = transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    
    model = Net()
    device = 'cuda'

    model.to(device)
    opt = optim.Adam(model.parameters(),lr=2e-4)
    losses = []
    images = []

    criterion = nn.BCELoss()

    num_epochs = 50
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            new_labels = torch.zeros((len(labels),10))
            for j in range(0,len(labels)):
                age = labels[j]
                if age <=45: 
                    new_labels[j][0] = 1
                if 45<age <= 50:
                    new_labels[j][1] = 1
                if 50<age <= 55:
                    new_labels[j][2] = 1
                if 55<age <= 60:
                    new_labels[j][3] = 1
                if 60<age <= 65:
                    new_labels[j][4] = 1
                if 65<age <= 70:
                    new_labels[j][5] = 1
                if 70<age <= 75:
                    new_labels[j][6] = 1
                if 75<age <= 80:
                    new_labels[j][7] = 1
                if 80<age <= 85:
                    new_labels[j][8] = 1
                if 85<age:
                    new_labels[j][9] = 1
                
            
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))

            loss = criterion(outputs, new_labels.to(device))
            #print("labels",labels.shape)
            #print("outputs",outputs.shape)
            #loss = criterion(torch.flatten(outputs).type(torch.FloatTensor),labels.type(torch.FloatTensor))
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
        print(epoch)

    PATH = './age_prediction.pth'
    torch.save(model.state_dict(), PATH)
    print('Finished Training')

    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.to(device))
            new_labels = torch.zeros((len(labels),10))
            for j in range(0,len(labels)):
                age = labels[j]
                if age <=45: 
                    new_labels[j][0] = 1
                if 45<age <= 50:
                    new_labels[j][1] = 1
                if 50<age <= 55:
                    new_labels[j][2] = 1
                if 55<age <= 60:
                    new_labels[j][3] = 1
                if 60<age <= 65:
                    new_labels[j][4] = 1
                if 65<age <= 70:
                    new_labels[j][5] = 1
                if 70<age <= 75:
                    new_labels[j][6] = 1
                if 75<age <= 80:
                    new_labels[j][7] = 1
                if 80<age <= 85:
                    new_labels[j][8] = 1
                if 85<age:
                    new_labels[j][9] = 1
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            _, labels_idx = torch.max(new_labels.data,1)
            total += labels.size(0)
            #print(new_labels.shape)
            #print(outputs.shape)
            #print(predicted.shape)
            #print(predicted)
            #print(torch.max(new_labels.data,1))
            correct += (predicted.to(device) == labels_idx.to(device)).sum().item()
            #correct += (predicted.to(device) == new_labels.to(device)).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')



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




    






