import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.3):
        super(Simple3DCNN, self).__init__()
        # Initial convolution with batch norm
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # /2
        )
        
        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64)
        )
        self.downsample1 = nn.Conv3d(32, 64, kernel_size=1, stride=1)
        
        # Pooling after residual
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # /4
        
        # Conv layers with dropout
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # /8
        
        # Scale-aware fully connected
        self.fc_scale = nn.Linear(1, 64)  # Processes scaling factor
        self.fc1 = nn.Linear(128 * 16 * 16 * 16 + 64, 512)  # + scale features
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x, scale_factor):
        # Input x: (B, 1, 128, 128, 128)
        x = self.conv1(x)  # -> (B, 32, 64, 64, 64)
        
        # Residual block
        residual = self.downsample1(x)
        x = F.relu(self.res1(x) + residual)
        x = self.pool2(x)  # -> (B, 64, 32, 32, 32)
        
        # Conv + dropout
        x = self.conv2(x)  # -> (B, 128, 32, 32, 32)
        x = self.pool3(x)  # -> (B, 128, 16, 16, 16)
        
        # Flatten and concatenate scale factor
        x_flat = x.view(x.size(0), -1)  # (B, 128*16*16*16)
        scale_feat = F.relu(self.fc_scale(scale_factor.unsqueeze(1)))  # (B, 64)
        x_combined = torch.cat([x_flat, scale_feat], dim=1)
        
        # Classifier
        x = F.relu(self.fc1(x_combined))
        x = self.fc2(x)
        return x