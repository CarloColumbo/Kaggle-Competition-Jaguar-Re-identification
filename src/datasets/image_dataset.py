from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    PyTorch Dataset for images.
    """

    def __init__(self,
                 image_paths,
                 labels=None,
                 transform_fn=None,
                 preprocess_fn=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform_fn = transform_fn
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Load image and convert to RGBA if alpha channel is used for background interventions
        img = Image.open(image_path).convert("RGBA")
        
        if self.transform_fn:
            img = self.transform_fn(img)

        img = img.convert("RGB")  # Convert back to RGB
        if self.preprocess_fn:
            img = self.preprocess_fn(img)
        
        if self.labels is None:
            return img
            
        label = self.labels[idx]
        return img, label
