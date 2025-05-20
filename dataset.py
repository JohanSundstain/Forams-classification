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

class TrainDataset(Dataset):
	def __init__(self, tiff_paths, labels=None):
		self.files_list = os.listdir(tiff_paths)
		self.samples = []

		with open(labels, "r") as f:
			self.labels = f.readlines()
		
		self.labels = self.labels[1::]
		for file_name, line in zip(self.files_list, self.labels):	
			path_to_tiff = os.path.join(self.tiff_paths, file_name)
			label = int(line.split(",")[1])
			self.samples.append( (path_to_tiff, label) )

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		volume_path, label = self.samples[idx]

		volume = load_tiff_stack_to_3d_array(volume_path)
		volume = preprocess_tiff(volume=volume)

		label = torch.tensor(label).long()

		return volume, label
	
class TrainWithValidDataset(Dataset):
	valid_samples = []
	train_samples = []
	samples = []
	def __init__(self, tiff_paths, labels, k=0.1):
		self.files_list = os.listdir(tiff_paths)
		self.k = k
		self.labels = {}

		valid_files, train_files = self._separate()
	
		with open(labels, "r") as f:
			raws = f.readlines()

		raws = raws[1::]		
		for raw in raws:
			# labelled_xxxxx, y -> [labelled_xxxxx, y]
			separated = raw.split(",")
			# labelled_xxxxx -> [labelled, xxxxx] -> int(xxxxx); 
			# y -> int y
			file_idx, class_idx = int(separated[0].split("_")[1]), int(separated[1])
			self.labels[file_idx] = class_idx


		for file_name in valid_files:	
			path_to_tiff = os.path.join(tiff_paths, file_name)

			# get file index
			# file_name = labelled_foram_xxxxx_sc_y_yyy.tif -> [labelled, foram, xxxxx, sc, y, yyy.tif] -> int(xxxxx)
			file_idx = int(file_name.split("_")[2])
			
			TrainWithValidDataset.valid_samples.append( (path_to_tiff, self.labels[file_idx]) )
		
		for file_name in train_files:	
			path_to_tiff = os.path.join(tiff_paths, file_name)

			# get file index
			# file_name = labelled_foram_xxxxx_sc_y_yyy.tif -> [labelled, foram, xxxxx, sc, y, yyy.tif] -> int(xxxxx)
			file_idx = int(file_name.split("_")[2])
			
			TrainWithValidDataset.train_samples.append( (path_to_tiff, self.labels[file_idx]) ) 
			
		TrainWithValidDataset.train()
		
	@staticmethod	
	def train():
		TrainWithValidDataset.samples = TrainWithValidDataset.train_samples
	
	@staticmethod
	def valid():
		TrainWithValidDataset.samples = TrainWithValidDataset.valid_samples

	@staticmethod
	def clear():
		TrainWithValidDataset.samples.clear()
		TrainWithValidDataset.train_samples.clear()
		TrainWithValidDataset.valid_samples.clear()
	

	def _separate(self):
		n = int(len(self.files_list) * self.k)
		shuffled = self.files_list.copy()
		random.shuffle(shuffled)
		return shuffled[:n], shuffled[n:] 

	def __len__(self):
		return len(TrainWithValidDataset.samples)

	def __getitem__(self, idx):
		volume_path, label  = TrainWithValidDataset.samples[idx]

		volume = load_tiff_stack_to_3d_array(volume_path)
		volume = preprocess_tiff(volume=volume)
		label = torch.tensor(label).long()

		return volume, label 
	

class TestDataset(Dataset):
	def __init__(self, tiff_paths):
		self.tiff_paths = tiff_paths
		files_list = os.listdir(tiff_paths)
		self.samples = []

		for file_name in files_list:	
			path_to_tiff = os.path.join(self.tiff_paths, file_name)
			file_idx = int(file_name.split("_")[1])
			self.samples.append( (path_to_tiff, file_idx) )

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		volume_path, file_idx  = self.samples[idx]

		volume = load_tiff_stack_to_3d_array(volume_path)
		volume = preprocess_tiff(volume=volume)

		file_idx = torch.tensor(file_idx).long()

		return volume, file_idx