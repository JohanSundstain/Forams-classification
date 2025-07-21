import os

import torch
import torch.nn as nn
import numpy as np
import tifffile
from torch.utils.data import  DataLoader
from lgbt import lgbt

from dataset import KFoldValidDataset 
from model import Enhanced3DCNN 

# ========================================
# Global variables
# ========================================
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
tiff_path =  "./dataset/volumes/volumes/labelled"
labels =  "./dataset/labelled.csv"
epochs = 200
lr = 0.0001
best_loss = float('inf')

if __name__ == "__main__":

	model = Enhanced3DCNN(num_classes=14)
	model.to(device=device)
	
	dataset = KFoldValidDataset(tiff_paths=tiff_path, labels=labels, k=5)
	dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-4)
	loss_func = nn.CrossEntropyLoss()
	
	for epoch in range(epochs):
		model.train()
		dataset.train()
	
		running_loss = 0.0
		for volume, label in lgbt(dataloader, desc=f"train {epoch}", mode="swe"):
			volume = volume.to(device)
			label = label.to(device)
				
			optimizer.zero_grad()
			output = model(volume)
			loss = loss_func(output, label)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		running_loss = running_loss / len(dataloader)
		print(f"loss {running_loss}")
	
		model.eval()
		dataset.valid()
		running_loss = 0.0
		with torch.no_grad():
			for volume, label in lgbt(dataloader, desc=f"valid {epoch}", mode="nor"):
				volume = volume.to(device)
				label = label.to(device)
				
				output = model(volume)
				loss = loss_func(output, label)
				running_loss += loss.item()

			running_loss = running_loss / len(dataloader)
			print(f"loss {running_loss}")

		if running_loss < best_loss:
			patience = 0
			best_loss = running_loss
			torch.save(model.state_dict(), f"weights/{type(model).__name__}{int(running_loss*100)}.pth")

	model.eval()
	torch.save(model.state_dict(), "weights/last.pth")
