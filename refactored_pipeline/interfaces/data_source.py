"""
Data source interface for training pipeline.

This interface defines the contract for data loading and processing components,
enabling different data sources to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Tuple
from torch.utils.data import Dataset, DataLoader


class IDataSource(ABC):
    """
    Interface for data sources that provide training and validation data.
    
    Implementations should handle data loading, preprocessing, tokenization,
    and dataset creation while being memory-efficient and scalable.
    """
    
    @abstractmethod
    def load_raw_data(self, config: Dict[str, Any]) -> Any:
        """
        Load raw data from the source.
        
        Args:
            config: Configuration dictionary containing data loading parameters
            
        Returns:
            Raw data object (e.g., HuggingFace dataset, file paths, etc.)
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, raw_data: Any, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Preprocess raw data into training and validation sets.
        
        Args:
            raw_data: Raw data from load_raw_data()
            config: Configuration dictionary containing preprocessing parameters
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        pass
    
    @abstractmethod
    def create_datasets(self, train_data: Any, val_data: Any, config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
        """
        Create PyTorch datasets from preprocessed data.
        
        Args:
            train_data: Preprocessed training data
            val_data: Preprocessed validation data
            config: Configuration dictionary containing dataset parameters
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        pass
    
    @abstractmethod
    def create_data_loaders(self, train_dataset: Dataset, val_dataset: Dataset, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders from datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration dictionary containing data loader parameters
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size for the tokenizer.
        
        Returns:
            Size of the vocabulary
        """
        pass
    
    @abstractmethod
    def tokenize_text(self, text: str) -> list:
        """
        Tokenize a text string.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode_tokens(self, tokens: list) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        pass
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary containing data statistics (optional implementation)
        """
        return {}
    
    def validate_data_integrity(self) -> bool:
        """
        Validate the integrity of loaded data.
        
        Returns:
            True if data is valid, False otherwise (optional implementation)
        """
        return True


class IDocumentIndexer(ABC):
    """
    Interface for document indexing components.
    
    This interface will be used for the markdown-indexer integration,
    providing document processing and indexing capabilities.
    """
    
    @abstractmethod
    def index_document(self, document_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Index a single document.
        
        Args:
            document_path: Path to the document to index
            metadata: Optional metadata to associate with the document
            
        Returns:
            Document ID in the index
        """
        pass
    
    @abstractmethod
    def search_documents(self, query: str, top_k: int = 10) -> list:
        """
        Search for documents using a query.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of matching documents with scores
        """
        pass
    
    @abstractmethod
    def update_index(self, document_id: str, new_content: str) -> None:
        """
        Update an existing document in the index.
        
        Args:
            document_id: ID of the document to update
            new_content: New content for the document
        """
        pass
    
    @abstractmethod
    def remove_document(self, document_id: str) -> None:
        """
        Remove a document from the index.
        
        Args:
            document_id: ID of the document to remove
        """
        pass
    
    @abstractmethod
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary containing index statistics
        """
        pass
