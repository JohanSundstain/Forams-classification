import os
import shutil

import torch
import torch.nn as nn
import numpy as np
import tifffile
from torch.utils.data import  DataLoader
from lgbt import lgbt

from dataset import  TestDataset
from model import Enhanced3DCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source_path = r".\forams-classification-2025\volumes\volumes\unlabelled"
dest_path = r".\new_labelled"


if __name__ == "__main__":

	os.makedirs("new_labelled", exist_ok=True)
	out_csv = open("new_labelled.csv", "w")
	out_csv.write('id,label\n')

	model = Enhanced3DCNN(num_classes=14)
	model.load_state_dict(torch.load("weights/best103.pth"))
	model.to(device=device)

	dataset = TestDataset(source_path)
	train_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

	results = []
	source_files = os.listdir(source_path)
	global_index = 0

	model.eval()
	with torch.no_grad():
		for volume, file_idx, path in lgbt(train_dataloader, mode='esp', hero='unicorn', desc="Separate"):
			volume = volume.to(device=device)
			output = model(volume)
			prob = torch.softmax(output.squeeze(), dim=0)
			idx = torch.argmax(prob).item()
			if prob[idx] > 0.95:
				# x -> 0000x
				string_index = f"{global_index:05d}"
				# (/home/opt/foram_0000y_sc_0_752.tif,) -> foram_0000y_sc_0_752.tif
				file_name = os.path.basename(path[0])
				# foram_00000_sc_0_752.tif -> [foram, 0000y, sc, 0, 752.tif]
				splited_file_name = file_name.split("_")
				# [foram, 0000y, sc, 0, 752.tif] -> [foram, 0000x, sc, 0, 752.tif]
				splited_file_name[1] = string_index
				# [foram, 0000x, sc, 0, 752.tif] -> labelled_foram_0000x_sc_0_752.tif
				new_file_name = "labelled_" + "_".join(splited_file_name)

				dest = os.path.join(dest_path, new_file_name)	
				shutil.copy(path[0], dest)

				out_csv.write(f'labelled_{string_index},{idx}\n')

				global_index += 1

	# =====================================
	# Part to add labelled to new_labelled
	# =====================================	
	labelled_path = r".\forams-classification-2025\volumes\volumes\labelled"
	files = os.listdir(labelled_path)

	out_csv.close()

	out_csv = open("new_labelled.csv", "a")

	with open(r".\forams-classification-2025\labelled.csv", "r") as f:
		lines = f.readlines()

	lines = lines[1::]
	for file_name, line in zip(files, lines):
		class_idx = int(line.split(",")[1])
		new_id = f"labelled_{n:05d}"
		new_raw = f'{new_id},{class_idx}\n'
		out_csv.write(new_raw)

		tokens = file_name.split("_")
		tokens[2] = f'{n:05d}'
		new_name = "_".join(tokens)
		source_path = os.path.join(labelled_path, file_name)
		dest_path = os.path.join(r".\new_labelled",new_name )
		print(source_path, dest_path)
		shutil.copy(source_path, dest_path)

		n+=1 
	out_csv.close()			