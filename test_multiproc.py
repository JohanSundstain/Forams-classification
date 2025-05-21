import os
import multiprocessing

import torch
import torch.nn as nn
import numpy as np
from lgbt import lgbt

from dataset import TestDataset
from model import Enhanced3DCNN


# ========================================
# Global variables
# ========================================
tiff_path = "forams-classification-2025/volumes/volumes/unlabelled"
model = None
device = None

def init_model():
	global model
	global device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Enhanced3DCNN(num_classes=14)
	model.load_state_dict(torch.load("./weights/best8.pth"))	
	model.to(device=device)
	model.eval()

def single_task(elem):
	global model, device

	volume, file_idx = elem
	volume = volume.unsqueeze(dim=0).to(device=device)

	with torch.no_grad():
		output = model(volume)
		prob = torch.softmax(output.squeeze(), dim=0)
		idx = torch.argmax(prob).item()
		if prob[idx] > 0.9:
			class_idx = idx
		else:
			class_idx = 14
	return (file_idx.item(), class_idx)


if __name__ == "__main__":
	all_files = []

	dataset = TestDataset(tiff_path)
	names = os.listdir(tiff_path)
	for name in names:
		path_to_file = os.path.join(tiff_path, name)
		all_files.append(path_to_file)

	with multiprocessing.Pool(processes=8, initializer=init_model) as pool:
		results = list(lgbt(pool.imap(single_task, dataset), mode="nor", desc="Files", total=len(all_files)))

	results.sort(key=lambda x: x[0])
	
	with open("output.csv", "w") as output_file:
		output_file.write("id,label\n")
		for file_idx, class_idx in results:
			output_file.write(f'{file_idx},{class_idx}\n')