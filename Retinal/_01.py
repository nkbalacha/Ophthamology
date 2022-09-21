import logging

import mne
import toml
from glob2 import glob
from tqdm import tqdm

from models.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def main():
	model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8)
	).cuda()

	diffusion = GaussianDiffusion(
    	model,
    	image_size = 1024,
    	timesteps = 10,           # number of steps
    	sampling_timesteps = 2,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    	loss_type = 'l1'            # L1 or L2
	).cuda()

	trainer = Trainer(
    	diffusion,
    	#'/home/nitin/data/resized_train_cropped/resized_train_cropped',
		_KAGGLE_DATA_DIR,
    	train_batch_size = 1,
    	train_lr = 8e-5,
    	train_num_steps = 100,         # total training steps
    	gradient_accumulate_every = 2,    # gradient accumulation steps
    	ema_decay = 0.995,                # exponential moving average decay
    	results_folder= _RESULTS_DIR,
		amp = True                        # turn on mixed precision
	)

	trainer.train()

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