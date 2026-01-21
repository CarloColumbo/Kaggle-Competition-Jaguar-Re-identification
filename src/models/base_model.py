import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models to make logging simpler.
    
    Args:
        None
        
    Returns:
        None
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_embedding_size(self) -> int:
        """
        Get the size of the embedding produced by the model.
        """
        pass
    
    @abstractmethod
    def get_fusion_strategy(self) -> str:
        """
        Get the fusion strategy used by the model.
        """
        pass
    
    def get_number_of_parameters(self) -> int:
        """
        Get the total number of parameters in the model.
        We also count non-trainable parameters.
        """
        return sum(p.numel() for p in self.parameters())
