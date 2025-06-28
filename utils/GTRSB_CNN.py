import torch.nn as nn
import torch.nn.functional as F

# The design of the residual block is referenced from:
# https://arxiv.org/pdf/1512.03385

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity # Add the shortcut connection
        out = self.relu(out)
        return out

# Reference:
# https://www.kaggle.com/code/kooaslansefat/cnn-97-accuracy-plus-safety-monitoring-safeml/notebook#Defining-the-CNN-Model

# The original model in Pytorch from the above link:

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=43):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout(p=0.25)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 4 * 4, 256)
#         self.dropout2 = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(256, num_classes)
        
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.dropout1(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.dropout1(x)
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # Replace conv2, bn2 with a ResidualBlock for deeper layers
        self.res_block1 = ResidualBlock(32, 64, stride=1)
        
        # Replace conv3, bn3 with another ResidualBlock
        self.res_block2 = ResidualBlock(64, 128, stride=1)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Input tensor shape: (batch_size, 3, 64, 64)

        # First convolution, batch norm, ReLU, and max pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x)))) # (batch_size, 32, 32, 32)
        
        # First residual block
        x = self.res_block1(x) # (batch_size, 64, 32, 32)
        # Max pooling
        x = self.pool(x) # (batch_size, 64, 16, 16)
        
        # Second residual block
        x = self.res_block2(x) # (batch_size, 128, 16, 16)
        # Max pooling
        x = self.pool(x) # (batch_size, 128, 8, 8)
        
        x = self.relu(self.bn4(self.conv4(x))) # (batch_size, 128, 10, 10) due to padding=2
        # Max pooling
        x = self.pool(x) # (batch_size, 128, 5, 5)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1) # (batch_size, 3200)
        
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x)) # (batch_size, 1024)
        
        # Output layer
        x = self.fc2(x) # (batch_size, num_classes)
        return x