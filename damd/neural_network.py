import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DAMD_CNN(nn.Module):

    def __init__(self, num_tokens, embedding_dim):
        super(DAMD_CNN, self).__init__()
        self.emb = nn.Embedding(num_tokens + 1, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, 8)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        output_emb = self.emb(x)
        output_emb = output_emb.permute(0, 2, 1)
        output_conv1 = F.relu(self.conv1(output_emb))
        output_gmp = F.adaptive_max_pool1d(output_conv1, output_size=1)
        output_fc1 = F.relu(self.fc1(output_gmp.squeeze()))
        output_fc2 = self.fc2(output_fc1)
        return output_fc2

    def get_embedding(self, x):
        return self.emb(x)

    def forward_embedding(self, embedding_input):
        output_emb = embedding_input.permute(0, 2, 1)
        output_conv1 = F.relu(self.conv1(output_emb))
        output_gmp = F.adaptive_max_pool1d(output_conv1, output_size=1)
        output_fc1 = F.relu(self.fc1(output_gmp.squeeze()))
        output_fc2 = self.fc2(output_fc1)
        return output_fc2
