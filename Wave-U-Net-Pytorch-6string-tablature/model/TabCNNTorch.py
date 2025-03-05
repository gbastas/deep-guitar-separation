import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adadelta
import torch.nn.init as init

class TabCNN(nn.Module):
    def __init__(self, num_strings=6, num_classes=21, input_shape=(192, 9, 1)):
        super(TabCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(5952, 128)  # now we use the correct number here
        self.fc2 = nn.Linear(128, num_classes * num_strings)
        self.num_strings = num_strings
        self.num_classes = num_classes

        # Initialize weights with Xavier/Glorot Uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = x.view(-1, self.num_strings, self.num_classes)
        return x  

