import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim = 4):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc4 = nn.Linear(hidden_dim[2], output_dim)
        #self.softmax = nn.Softmax(dim=1)  #softmax for several classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
  
        #output = self.softmax(output)  
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

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
