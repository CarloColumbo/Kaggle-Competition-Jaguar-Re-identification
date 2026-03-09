import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BlurImageDataset(Dataset):
    """
    PyTorch Dataset for images.
    This dataset applies a Gaussian blur to the background (where alpha=0).
    """
    
    def __init__(self, filepaths, label=None, preprocess_fn=None):
        self.filepaths = filepaths
        self.labels = label
        self.preprocess_fn = preprocess_fn

    def blur_background(self, img):
        img = np.array(img.convert("RGBA"))

        if img.shape[2] != 4:
            raise ValueError("Input image must have 4 channels (RGBA)")

        rgb = img[:, :, :3]
        alpha = img[:, :, 3]

        blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=10, sigmaY=10)

        mask = alpha == 0
        rgb = rgb.copy()
        rgb[mask] = blurred[mask]

        return Image.fromarray(rgb)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGBA")
        image = self.blur_background(image)
        
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)
            
        if self.labels is None:
            return image

        label = self.labels[idx]
        return image, label
