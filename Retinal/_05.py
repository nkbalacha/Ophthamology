import logging

import mne
import toml
from glob2 import glob
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import sys
import math
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from models.diff_pytorch import DiffusionModel
from torchvision import transforms
from torchvision.io import read_image

def main():
    batch_size = 512
    resize = 128
    diff_steps = 1000
    img_depth = 3
    max_epoch = 100
    pass_version = None
    last_checkpoint = None
    transform = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Grayscale(),
                transforms.Resize((resize,resize)),
                #transforms.ToTensor(),
                transforms.ToTensor()
                #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
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
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(trainset, batch_size=batch_size//16, shuffle=True)

    model = DiffusionModel(resize*resize, diff_steps, img_depth)

    tb_logger = pl.loggers.TensorBoardLogger(
    _OUTPUT_DIR +"lightning_logs/",
    name="kaggle",
    version=None,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch, 
        log_every_n_steps=10, 
        gpus=1, 
        auto_select_gpus=True,
        resume_from_checkpoint=last_checkpoint, 
        logger=tb_logger
    )
    trainer.fit(model, train_loader, val_loader)
    #trainer.fit(model, train_loader)

    gif_shape = [3, 3]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((sample_batch_size, img_depth, resize, resize))
    sample_steps = torch.arange(model.t_range-1, 0, -1)
    for t in sample_steps:
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2


    # Process samples and save as gif
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], resize, resize, img_depth)

    def stack_samples(gen_samples, stack_dim):
        gen_samples = list(torch.split(gen_samples, 1, dim=1))
        for i in range(len(gen_samples)):
            gen_samples[i] = gen_samples[i].squeeze(1)
        return torch.cat(gen_samples, dim=stack_dim)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    imageio.mimsave(
        f"{trainer.logger.log_dir}/pred.gif",
        list(gen_samples),
        fps=5,
    )

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