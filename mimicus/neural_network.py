import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MIMICUS_MLP(nn.Module):

    def __init__(self, num_features):
        super(MIMICUS_MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, 200)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        output_fc1 = F.relu(self.dropout1(self.fc1(x)))
        output_fc2 = F.relu(self.dropout2(self.fc2(output_fc1)))
        output_fc3 = self.fc3(output_fc2)
        return output_fc3
