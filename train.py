import os

import torch
import torch.nn as nn
import numpy as np
import tifffile
from torch.utils.data import  DataLoader
from lgbt import lgbt

from dataset import TrainWithValidDataset, TrainDataset 
from model import Enhanced3DCNN, Enhanced3DCNN_new

# ========================================
# Global variables
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tiff_path =  "./forams-classification-2025/volumes/volumes/labelled"
labels =  "./forams-classification-2025/labelled.csv"
epochs = 200
lr = 0.0001
best_loss = float('inf')
patience = 0

if __name__ == "__main__":

	for i in range(10):
		best_loss = float('inf')
		patience = 0
		model = Enhanced3DCNN_new(num_classes=14)
		model.to(device=device)
	
		dataset = TrainWithValidDataset(tiff_paths=tiff_path, labels=labels, k=0.1)
		dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
	
		optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-5)
		loss_func = nn.CrossEntropyLoss()
	
		for epoch in range(epochs):
			model.train()
			dataset.train()
	
			running_loss = 0.0
			for volume, label in lgbt(dataloader, desc=f"train14 {epoch}", mode="swe"):
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
			patience += 1
			with torch.no_grad():
				for volume, label in lgbt(dataloader, desc=f"valid14 {epoch}", mode="nor"):
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
				torch.save(model.state_dict(), f"weights/{type(model).__name__}best_14_class{int(running_loss*100)}.pth")
			
			if patience == 10:
				print("No changes, swap dataset")
				dataset.clear()
				break
	
		model.eval()
		torch.save(model.state_dict(), "weights/last.pth")
