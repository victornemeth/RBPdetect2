"""Model architecture for the RBP classifier: a linear head on frozen PLM embeddings."""

import torch.nn as nn


class LinearClassifier(nn.Module):
    """Single linear layer mapping frozen PLM embeddings to class logits."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.head(x)
