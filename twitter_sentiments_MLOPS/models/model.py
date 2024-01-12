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


