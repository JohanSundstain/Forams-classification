import os
import random

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader


def load_tiff_stack_to_3d_array(tiff_path):
	"""Load a multi-page TIFF stack into a 3D numpy array (D, H, W)."""
	with tifffile.TiffFile(tiff_path) as tif:
		stack = tif.asarray().astype(np.float32)
	return stack

def preprocess_tiff(volume):
	volume = (volume - volume.min()) / (volume.max() - volume.min())
	volume = torch.from_numpy(volume).unsqueeze(dim=0)
	
	return volume

class KFoldValidDataset(Dataset):
	valid_samples = []
	train_samples = []
	samples = []
	def __init__(self, tiff_paths, labels, k=3):
		self._current_fold = -1
		self._k = k
		self._train = True

		files_list = os.listdir(tiff_paths)
		with open(labels, "r") as f:
			raws = f.readlines()
		raws = raws[1::]	
		for raw, file_name in zip(raws, files_list):
			class_idx = int(raw.split(",")[1])
			file_path = os.path.join(tiff_paths, file_name)
			KFoldValidDataset.samples.append( (class_idx, file_path) )
		
		random.shuffle(KFoldValidDataset.samples)
		self._num_val_samples = len(KFoldValidDataset.samples) // self._k
		self.next_fold()

		
	def train(self):
		self._train = True
	
	def valid(self):
		self._train = False

	def clear():
		KFoldValidDataset.train_samples.clear()
		KFoldValidDataset.valid_samples.clear()

	def next_fold(self):
		self._current_fold += 1

		KFoldValidDataset.valid_samples = KFoldValidDataset.samples[self._current_fold * self._num_val_samples : (self._current_fold + 1) * self._num_val_samples]
		KFoldValidDataset.train_samples = KFoldValidDataset.samples[:self._current_fold * self._num_val_samples] + KFoldValidDataset.samples[(self._current_fold + 1) * self._num_val_samples:] 
		
	
	def __len__(self):
		return len(KFoldValidDataset.train_samples if self._train else KFoldValidDataset.valid_samples)

	def __getitem__(self, idx):
		label, volume_path = KFoldValidDataset.train_samples[idx] if self._train else  KFoldValidDataset.valid_samples[idx]

		volume = load_tiff_stack_to_3d_array(volume_path)
		volume = preprocess_tiff(volume=volume)
		label = torch.tensor(label).long()

		return volume, label 
	