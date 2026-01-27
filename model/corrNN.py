import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(CorrNN, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        Returns: Tensor of shape (batch_size, num_classes)
        """
        x = self.fc(x)
        x = self.bn(x)
        # out = F.log_softmax(x, dim=1)  # or use nn.CrossEntropyLoss() with raw logits
        return x
