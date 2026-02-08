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
    
    def __init__(self):
        super().__init__()
        
        self.megadescriptor = timm.create_model(
            "hf-hub:BVRA/MegaDescriptor-L-384",
            pretrained=True
        )
        self.megadescriptor.eval()
        self.megadescriptor
        
    def to(self, device):
        self.megadescriptor.to(device)
        return self
        
    def forward(self, x):
        return self.megadescriptor(x)

    def get_embedding_size(self) -> int:
        return 1536
    
    @staticmethod
    def get_model_name() -> str:
        return "hf-hub:BVRA/MegaDescriptor-L-384"
        
    @staticmethod
    def get_input_size() -> int:
        return 384
