import torch
from torch.utils.data import Dataset
import numpy as np


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for pre-computed embeddings.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        
        assert len(self.embeddings) == len(self.labels), "Embeddings and labels must have the same length"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
