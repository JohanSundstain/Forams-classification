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



class Enhanced3DCNN_new(nn.Module):
	def __init__(self, num_classes=14, dropout_prob=0.3):
		super(Enhanced3DCNN_new, self).__init__()
		
		# Input: (B, 1, 128, 128, 128)
		self.features = nn.Sequential(
			# Initial Block
			nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=True),
			nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(2),  # -> (32, 64, 64, 64)

			# Bottleneck Block 1
			self._make_bottleneck_block(32, 64),
			nn.MaxPool3d(2),  # -> (64, 32, 32, 32)
			
			# Bottleneck Block 2
			self._make_bottleneck_block(64, 128),
			nn.MaxPool3d(2),  # -> (128, 16, 16, 16)
			
			# Bottleneck Block 3
			self._make_bottleneck_block(128, 256),
			nn.MaxPool3d(2),  # -> (256, 8, 8, 8)
			
			# Final compression
			nn.Conv3d(256, 512, kernel_size=1, bias=True),
			nn.InstanceNorm3d(512, affine=True),
			nn.ReLU(inplace=True),
			
			# Adaptive pooling
			nn.AdaptiveAvgPool3d((4, 4, 4)))
		
		# Classifier head
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(512 * 4 * 4 * 4, 1024, bias=True),
			nn.LayerNorm(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout_prob),
			
			nn.Linear(1024, 512, bias=True),
			nn.LayerNorm(512),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout_prob/2),
			
			nn.Linear(512, num_classes, bias=True))
		
		# Initialize weights
		self._initialize_weights()

	def _make_bottleneck_block(self, in_channels, out_channels):
		return nn.Sequential(
			# Compression
			nn.Conv3d(in_channels, in_channels//4, kernel_size=1, bias=True),
			nn.InstanceNorm3d(in_channels//4, affine=True),
			nn.ReLU(inplace=True),
			
			# Spatial processing
			nn.Conv3d(in_channels//4, in_channels//4, kernel_size=3, padding=1, bias=True),
			nn.InstanceNorm3d(in_channels//4, affine=True),
			nn.ReLU(inplace=True),
			
			# Expansion
			nn.Conv3d(in_channels//4, out_channels, kernel_size=1, bias=True),
			nn.InstanceNorm3d(out_channels, affine=True),
			nn.ReLU(inplace=True)
		)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.InstanceNorm3d) or isinstance(m, nn.LayerNorm):
				if m.weight is not None:
					nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x