"""
Model interface for training pipeline.

This interface defines the contract for model components,
enabling different model architectures to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class IModel(ABC):
    """
    Interface for model components in the training pipeline.
    
    Implementations should provide forward pass, parameter management,
    and optimization configuration capabilities.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (typically logits)
        """
        pass
    
    @abstractmethod
    def get_num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.
        
        Returns:
            Total number of parameters
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        pass
    
    @abstractmethod
    def configure_optimizers(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Configure and return an optimizer for the model.
        
        Args:
            config: Configuration dictionary containing optimizer parameters
            
        Returns:
            Configured optimizer
        """
        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: str, additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            additional_data: Additional data to save with the model
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        pass
    
    def enable_compilation(self) -> bool:
        """
        Enable model compilation for performance optimization.
        
        Returns:
            True if compilation was successful, False otherwise
        """
        return False  # Default implementation
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics for the model.
        
        Returns:
            Dictionary containing memory usage information
        """
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "cached_mb": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
            }
        return {"allocated_mb": 0, "cached_mb": 0, "max_allocated_mb": 0}
    
    def reset_memory_stats(self) -> None:
        """Reset memory usage statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class IAttentionStrategy(ABC):
    """
    Interface for attention mechanisms.
    
    This allows different attention implementations to be used
    within models (e.g., standard attention, flash attention, etc.)
    """
    
    @abstractmethod
    def compute_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention scores and apply to values.
        
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output tensor
        """
        pass
    
    @abstractmethod
    def get_attention_config(self) -> Dict[str, Any]:
        """
        Get attention mechanism configuration.
        
        Returns:
            Configuration dictionary
        """
        pass


class IMLPStrategy(ABC):
    """
    Interface for MLP/feed-forward strategies.
    
    This allows different MLP implementations to be used
    within transformer blocks.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_mlp_config(self) -> Dict[str, Any]:
        """
        Get MLP configuration.
        
        Returns:
            Configuration dictionary
        """
        pass


class IModelFactory(ABC):
    """
    Interface for model factories.
    
    Factories handle model creation and configuration,
    allowing for different model types to be created based on configuration.
    """
    
    @abstractmethod
    def create_model(self, config: Dict[str, Any]) -> IModel:
        """
        Create a model instance based on configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Configured model instance
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> list:
        """
        Get list of supported model types.
        
        Returns:
            List of supported model type names
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
