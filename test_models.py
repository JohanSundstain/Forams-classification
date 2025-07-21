import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from lgbt import lgbt

from dataset import KFoldValidDataset 
from model import Enhanced3DCNN, Enhanced3DCNN_new

# ========================================
# Global variables
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tiff_path =  "./dataset/volumes/volumes/labelled"
labels =  "./dataset/labelled.csv"
epochs = 200
lr = 0.01
best_loss = float('inf')

def F1_score(y_true, y_pred):

	pass

if __name__ == "__main__":

	model = Enhanced3DCNN_new(num_classes=14)
	model.to(device=device)
	
	dataset = KFoldValidDataset(tiff_paths=tiff_path, labels=labels, k=5)
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-4)
	loss_func = nn.CrossEntropyLoss()

	for k in range(5):
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
		dataset.next_fold()

