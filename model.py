import torch
import torch.nn as nn
import torch.nn.functional as F

class Enhanced3DCNN(nn.Module):
	def __init__(self, num_classes=14, dropout_prob=0.3):
		super(Enhanced3DCNN, self).__init__()
		# Input: (B, 1, 128, 128, 128)
		self.features = nn.Sequential(
			# Block 1
			nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(2),	  # -> (32, 64, 64, 64)

			# Block 2
			nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(2),	  # -> (64, 32, 32, 32)

			# Block 3
			nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(2),	  # -> (128, 16, 16, 16)

			# Block 4
			nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm3d(256, affine=True),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(2),	  # -> (256, 8, 8, 8)

			# Adaptive pooling to reduce spatial dims
			nn.AdaptiveAvgPool3d((4, 4, 4))
		)
		
		# Classifier head
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(256 * 4 * 4 * 4, 512, bias=False),
			nn.LayerNorm(512),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout_prob),
			nn.Linear(512, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x
