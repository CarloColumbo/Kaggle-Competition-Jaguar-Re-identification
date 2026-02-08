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
        
    def get_embeddings(self, x):
        """
        Forward pass to get embeddings from input data.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output embeddings.
        """
        return self.forward(x)

    @abstractmethod
    def get_embedding_size(self) -> int:
        """
        Get the size of the embedding produced by the model.
        """
        pass
    
    def get_number_of_parameters(self) -> int:
        """
        Get the total number of parameters in the model.
        We also count non-trainable parameters.
        """
        return sum(p.numel() for p in self.parameters())
        
    def print_model_summary(self) -> None:
        """
        Print a summary of the model architecture and number of parameters.
        """
        print(f"Model loaded successfully")
        print(f"  Parameters: {self.get_number_of_parameters():,}")
        print(f"  Embedding dimension: {self.get_embedding_size()}")

    @staticmethod
    def get_model_name() -> str:
        """
        Get the name of the model.
        """
        return ""
        
    @staticmethod
    def get_input_size() -> int:
        """
        Get the expected input size of the model.
        """
        return 0
