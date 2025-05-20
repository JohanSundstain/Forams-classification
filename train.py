import os

import torch
import torch.nn as nn
import numpy as np
import tifffile
from torch.utils.data import  DataLoader
from lgbt import lgbt

from dataset import TiffVolumeDataset
from model import Simple3DCNN

# ========================================
# Global variables
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tiff_path = "./forams-classification-2025/volumes/volumes/labelled"
labels = "./forams-classification-2025/labelled.csv"
epochs = 100
lr = 0.0001
best_loss = float('inf')

if __name__ == "__main__":

	model = Simple3DCNN(num_classes=14)
	model.to(device=device)

	dataset = TiffVolumeDataset(tiff_path, labels)
	train_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	loss_func = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		for volume, label, scale_factor in lgbt(train_dataloader, desc=f"epoch {epoch}", mode="mex"):
			volume = volume.to(device)
			label = label.to(device)
			
			optimizer.zero_grad()
			output = model(volume, scale_factor)
			loss = loss_func(output, label)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		running_loss = running_loss / len(train_dataloader)
		print(f"loss {running_loss}")
		if running_loss < best_loss:
			best_loss = running_loss
			model.eval()
			torch.save(model.state_dict(), f"best{int(running_loss*100)}.pth")


	model.eval()
	torch.save(model.state_dict(), "last.pth")
