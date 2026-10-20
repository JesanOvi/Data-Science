import warnings
warnings.filterwarnings("ignore", category=Warning)
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class SimpleANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(13, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        logits = self.output(x)

        return logits