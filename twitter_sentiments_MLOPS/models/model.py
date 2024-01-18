import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class FCNN_model(nn.Module):
    def __init__(self):
        super(FCNN_model, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def get_embed(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        # Assuming input size of [batch_size, 768]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 192, 100)  # Adjust the input features of fc1 depending on the output of the last pooling layer
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
