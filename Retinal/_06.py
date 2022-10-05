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

import tensorfn
from torch.utils import data
from tensorfn import load_arg_config, load_wandb
from tensorfn import distributed as dist
from tensorfn.optim import lr_scheduler
from tqdm import tqdm


from diff_folder.model import UNet
from diff_folder.diffusion import GaussianDiffusion, make_beta_schedule
from diff_folder.config import DiffusionConfig

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)

def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def train(conf, loader, model, ema, diffusion, optimizer, scheduler, device, wandb):
    loader = sample_data(loader)

    pbar = range(conf.training.n_iter + 1)

    if dist.is_primary():
        pbar = tqdm(pbar, dynamic_ncols=True)

    for i in pbar:
        epoch, img = next(loader)
        img = img.to(device)
        time = torch.randint(
            0,
            conf.diffusion.beta_schedule["n_timestep"],
            (img.shape[0],),
            device=device,
        )
        loss = diffusion.p_loss(model, img, time)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
        )

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"epoch: {epoch}; loss: {loss.item():.4f}; lr: {lr:.5f}"
            )

            if wandb is not None and i % conf.evaluate.log_every == 0:
                wandb.log({"epoch": epoch, "loss": loss.item(), "lr": lr}, step=i)

            if i % conf.evaluate.save_every == 0:
                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    f"checkpoint/diffusion_{str(i).zfill(6)}.pt",
                )

def main():
    conf = load_arg_config(DiffusionConfig)
    batch_size = 512
    resize = 128
    diff_steps = 1000
    img_depth = 3
    max_epoch = 100
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
    wandb = None
    if dist.is_primary() and conf.evaluate.wandb:
        wandb = load_wandb()
        wandb.init(project="denoising diffusion")

    device = "cuda"
    beta_schedule = "linear"

    conf.distributed = dist.get_world_size() > 1

    model = conf.model.make()
    model = model.to(device)
    ema = conf.model.make()
    ema = ema.to(device)

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        if conf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)

    train(
        conf, train_loader, model, ema, diffusion, optimizer, scheduler, device, wandb
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