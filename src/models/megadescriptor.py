import torch
import torch.nn as nn
from torchvision import transforms
import timm

from src.models.base_model import BaseModel


class MegaDescriptorL384(BaseModel):
    """
    MegaDescriptor model for extracting jaguar embeddings.
    Uses a pre-trained MegaDescriptor backbone.
    """
    
    def __init__(self, config, device):
        super().__init__()
        
        self.megadescriptor = timm.create_model(
            "hf-hub:BVRA/MegaDescriptor-L-384",
            pretrained=True
        )
        self.megadescriptor.eval()
        self.megadescriptor.to(device)
        
        # MegaDescriptor expects 384x384 images normalized with ImageNet statistics
        self._preprocess = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
    def forward(self, x):
        return self.megadescriptor(x)

    def get_embedding_size(self) -> int:
        return 1536
        
    def preprocess(self, x):
        return self._preprocess(x)
    
    @staticmethod
    def get_model_name() -> str:
        return "hf-hub:BVRA/MegaDescriptor-L-384"
        
    @staticmethod
    def get_input_size() -> int:
        return 384
