import os

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader


def load_tiff_stack_to_3d_array(tiff_path):
    """Load a multi-page TIFF stack into a 3D numpy array (D, H, W)."""
    with tifffile.TiffFile(tiff_path) as tif:
        stack = tif.asarray().astype(np.float32)
    return stack


# ==============================================
# Create a PyTorch Dataset for batches
# ==============================================
class TiffVolumeDataset(Dataset):
	def __init__(self, tiff_paths, labels):
		self.tiff_paths = tiff_paths
		files_list = os.listdir(tiff_paths)
		self.samples = []

		with open(labels, "r") as f:
			self.labels = f.readlines()
		
		self.labels = self.labels[1::]

		for file_name, line in zip(files_list, self.labels):	
			path_to_tiff = os.path.join(self.tiff_paths, file_name)
			label = int(line.split(",")[1])
			self.samples.append( (path_to_tiff, label) )

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		volume_path, label = self.samples[idx]

		volume = load_tiff_stack_to_3d_array(volume_path)

		volume = (volume - volume.min()) / (volume.max() - volume.min())
		volume = torch.from_numpy(volume)
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
		volume_path, file_idx = self.samples[idx]

		volume = load_tiff_stack_to_3d_array(volume_path)

		volume = (volume - volume.min()) / (volume.max() - volume.min())
		volume = torch.from_numpy(volume)
		file_idx = torch.tensor(file_idx).long()

		return volume, file_idx