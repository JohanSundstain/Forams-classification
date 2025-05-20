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
	def __init__(self, tiff_paths, labels=None):
		self.tiff_paths = tiff_paths
		files_list = os.listdir(tiff_paths)
		self.samples = []

		if labels is None:
			for file_name in files_list:	
				path_to_tiff = os.path.join(self.tiff_paths, file_name)
				file_name_splits = file_name.replace(".tif","").split("_")
				file_idx = int(file_name_splits[2])
				scale = float(f"{file_name_splits[-2]}.{file_name_splits[-1]}")
				self.samples.append( (path_to_tiff, file_idx, scale) )
		else:
			with open(labels, "r") as f:
				self.labels = f.readlines()
		
			self.labels = self.labels[1::]
			for file_name, line in zip(files_list, self.labels):	
				path_to_tiff = os.path.join(self.tiff_paths, file_name)
				label = int(line.split(",")[1])
				file_name_splits = file_name.replace(".tif","").split("_")
				scale_factor =  float(f"{file_name_splits[-2]}.{file_name_splits[-1]}")
				self.samples.append( (path_to_tiff, label, scale_factor) )


	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		volume_path, label, scale_factor = self.samples[idx]

		volume = load_tiff_stack_to_3d_array(volume_path)

		volume = (volume - volume.min()) / (volume.max() - volume.min())
		volume = torch.from_numpy(volume).unsqueeze(dim=0)
		label = torch.tensor(label).long()
		scale_factor = torch.tensor(scale_factor).float()

		return volume, label, scale_factor

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