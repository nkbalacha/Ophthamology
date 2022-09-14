import logging

import mne
import toml
from glob2 import glob
from tqdm import tqdm



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