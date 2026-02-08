import torch
from torch.utils.data import Dataset
import numpy as np


class ImageDataset(Dataset):
    """
    PyTorch Dataset for images.
    """

    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        return img, label
