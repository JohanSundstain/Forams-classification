import os

import torch
import torch.nn as nn
import numpy as np
import tifffile
from torch.utils.data import  DataLoader
from lgbt import lgbt

from dataset import  TestDataset
from model import Enhanced3DCNN

# ========================================
# Global variables
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tiff_path = r".\forams-classification-2025\volumes\volumes\unlabelled"

if __name__ == "__main__":

	model = Enhanced3DCNN(num_classes=14)
	model.load_state_dict(torch.load("weights/best103.pth"))
	model.to(device=device)

	dataset = TestDataset(tiff_path)
	train_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
	results = []

	model.eval()
	with torch.no_grad():
		for volume, file_idx in lgbt(train_dataloader, mode='esp', hero='teddy', desc="Test"):
			volume = volume.to(device=device)
			output = model(volume)
			prob = torch.softmax(output.squeeze(), dim=0)
			idx = torch.argmax(prob).item()
			if prob[idx] > 0.9:
				class_idx = idx
				results.append((file_idx.item(), class_idx))

	results.sort(key=lambda x: x[0])
	
	with open("output.csv", "w") as output_file:
		output_file.write("id,label\n")
		for file_idx, class_idx in results:
			output_file.write(f'{file_idx},{class_idx}\n')
