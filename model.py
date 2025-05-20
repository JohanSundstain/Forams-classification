import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Simple3DCNN, self).__init__()
        
        # Input shape: (batch, channels, depth, height, width) = (B, 1, 128, 128, 128)
        
        # Conv3D layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, 128, 128, 128)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)                 # Output: (32, 64, 64, 64)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, 64, 64, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)                  # Output: (64, 32, 32, 32)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1) # Output: (128, 32, 32, 32)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)                  # Output: (128, 16, 16, 16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Apply 3D convolutions + ReLU + pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 16 * 16 * 16)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
