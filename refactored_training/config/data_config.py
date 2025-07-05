"""
Data configuration for ChemLLM training using HuggingFace integration.

This module provides configuration classes for data loading, preprocessing,
and validation using HuggingFace Datasets.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Designed to work seamlessly with HuggingFace Datasets and DataLoaders.
    """
    
    # Dataset configuration
    dataset_name: str = "iAli61/chempile-education-dedup-dedup"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    streaming: bool = False
    cache_dir: Optional[str] = None
    
    # Text processing
    max_length: int = 256
    stride: Optional[int] = None  # None means non-overlapping chunks
    min_length: int = 10  # Filter out very short sequences
    
    # Tokenization
    tokenizer_name: Optional[str] = None  # Will use model tokenizer if None
    add_special_tokens: bool = True
    padding: str = "max_length"  # "max_length", "longest", or False
    truncation: bool = True
    return_attention_mask: bool = True
    
    # Data splitting
    train_split_ratio: float = 0.9
    validation_split_ratio: float = 0.1
    test_split_ratio: float = 0.0
    split_seed: int = 42
    
    # Performance optimization
    num_proc: Optional[int] = None  # Number of processes for mapping
    batch_size: int = 1000  # Batch size for dataset mapping
    remove_columns: Optional[List[str]] = None
    
    # Memory optimization
    max_samples: Optional[int] = None  # Limit dataset size for testing
    shuffle_seed: Optional[int] = 42
    
    # Validation
    validate_data: bool = True
    min_chars_per_sample: int = 50
    max_chars_per_sample: int = 10000
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.stride is None:
            self.stride = self.max_length  # Non-overlapping by default
            
        # Ensure splits sum to 1.0
        total_split = self.train_split_ratio + self.validation_split_ratio + self.test_split_ratio
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
            
        # Validate max_length
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
            
        # Set default cache directory
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "chemllm")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass  
class DataValidationConfig:
    """Configuration for data validation and quality checks."""
    
    # Content validation
    min_text_length: int = 10
    max_text_length: int = 50000
    check_encoding: bool = True
    allowed_encodings: List[str] = field(default_factory=lambda: ["utf-8", "ascii"])
    
    # Chemical domain validation
    check_chemical_content: bool = True
    min_chemical_terms_ratio: float = 0.1  # Minimum ratio of chemical terms
    chemical_keywords: List[str] = field(default_factory=lambda: [
        "molecule", "chemical", "compound", "reaction", "element", "atom",
        "bond", "synthesis", "catalyst", "polymer", "organic", "inorganic"
    ])
    
    # Quality checks
    check_duplicates: bool = True
    duplicate_threshold: float = 0.95  # Similarity threshold for duplicates
    check_language: bool = True
    expected_language: str = "en"
    
    # Statistical validation
    compute_statistics: bool = True
    log_sample_texts: bool = True
    sample_size: int = 100


def create_data_config(
    model_name: str = "gpt2",
    dataset_name: str = "iAli61/chempile-education-dedup-dedup", 
    max_length: int = 256,
    batch_size: int = 4,
    **kwargs
) -> DataConfig:
    """
    Create data configuration with sensible defaults.
    
    Args:
        model_name: Model name to determine tokenizer
        dataset_name: HuggingFace dataset name  
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        **kwargs: Additional configuration options
        
    Returns:
        DataConfig instance
    """
    return DataConfig(
        dataset_name=dataset_name,
        tokenizer_name=model_name,  # Use model's tokenizer
        max_length=max_length,
        batch_size=batch_size,
        **kwargs
    )


def create_memory_efficient_config(
    dataset_name: str = "iAli61/chempile-education-dedup-dedup",
    max_length: int = 256,
    max_samples: int = 10000,
    **kwargs
) -> DataConfig:
    """
    Create memory-efficient configuration for testing/development.
    
    Args:
        dataset_name: HuggingFace dataset name
        max_length: Maximum sequence length  
        max_samples: Limit number of samples
        **kwargs: Additional configuration options
        
    Returns:
        DataConfig instance optimized for memory efficiency
    """
    return DataConfig(
        dataset_name=dataset_name,
        max_length=max_length,
        max_samples=max_samples,
        streaming=True,  # Use streaming for memory efficiency
        num_proc=1,  # Single process to reduce memory
        batch_size=100,  # Smaller batches
        **kwargs
    )


def create_distributed_config(
    dataset_name: str = "iAli61/chempile-education-dedup-dedup",
    max_length: int = 512, 
    world_size: int = 1,
    **kwargs
) -> DataConfig:
    """
    Create configuration optimized for distributed training.
    
    Args:
        dataset_name: HuggingFace dataset name
        max_length: Maximum sequence length
        world_size: Number of distributed processes
        **kwargs: Additional configuration options
        
    Returns:
        DataConfig instance optimized for distributed training
    """
    return DataConfig(
        dataset_name=dataset_name,
        max_length=max_length,
        num_proc=max(1, world_size * 2),  # More processes for distributed
        batch_size=1000,  # Larger batches for efficiency
        cache_dir=None,  # Let each process use its own cache
        **kwargs
    )
