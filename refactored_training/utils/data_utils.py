"""
Data utilities for ChemLLM using HuggingFace Datasets.

This module provides functions to load, preprocess, and validate datasets
using HuggingFace's optimized data loading infrastructure.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Iterator
from pathlib import Path

from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from torch.utils.data import DataLoader

from ..config.data_config import DataConfig, DataValidationConfig

logger = logging.getLogger(__name__)


def load_chempile_dataset(
    config: DataConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> DatasetDict:
    """
    Load and preprocess the ChemPILE dataset using HuggingFace Datasets.
    
    Args:
        config: Data configuration
        tokenizer: Optional tokenizer (will load from config if None)
        
    Returns:
        DatasetDict with train/validation splits
    """
    logger.info(f"Loading dataset: {config.dataset_name}")
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer_name = config.tokenizer_name or "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    try:
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.dataset_split,
            streaming=config.streaming,
            cache_dir=config.cache_dir
        )
        logger.info(f"Loaded dataset with {len(dataset) if not config.streaming else 'streaming'} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset {config.dataset_name}: {e}")
        raise
    
    # Limit dataset size if specified
    if config.max_samples and not config.streaming:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        logger.info(f"Limited dataset to {len(dataset)} samples")
    
    # Validate data if requested
    if config.validate_data:
        validation_config = DataValidationConfig()
        validate_dataset_quality(dataset, validation_config)
    
    # Preprocess text
    dataset = preprocess_text_data(dataset, config)
    
    # Tokenize dataset
    dataset = tokenize_dataset(dataset, tokenizer, config)
    
    # Create train/validation splits
    dataset_dict = create_train_val_splits(dataset, config)
    
    logger.info("Dataset preprocessing completed")
    return dataset_dict


def preprocess_text_data(dataset: Dataset, config: DataConfig) -> Dataset:
    """
    Preprocess text data with cleaning and filtering.
    
    Args:
        dataset: Input dataset
        config: Data configuration
        
    Returns:
        Preprocessed dataset
    """
    logger.info("Preprocessing text data")
    
    def clean_text(example):
        """Clean and normalize text."""
        text = example.get("text", "")
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Filter by length
        if len(text) < config.min_chars_per_sample or len(text) > config.max_chars_per_sample:
            return {"text": ""}  # Mark for removal
            
        return {"text": text}
    
    # Apply preprocessing
    dataset = dataset.map(
        clean_text,
        num_proc=config.num_proc,
        batch_size=config.batch_size,
        desc="Cleaning text"
    )
    
    # Filter out empty texts
    original_size = len(dataset) if not config.streaming else "unknown"
    dataset = dataset.filter(
        lambda x: len(x["text"]) >= config.min_length,
        num_proc=config.num_proc,
        desc="Filtering short texts"
    )
    
    if not config.streaming:
        logger.info(f"Filtered dataset: {original_size} -> {len(dataset)} samples")
    
    return dataset


def tokenize_dataset(
    dataset: Dataset, 
    tokenizer: PreTrainedTokenizer, 
    config: DataConfig
) -> Dataset:
    """
    Tokenize dataset using sliding window approach.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        config: Data configuration
        
    Returns:
        Tokenized dataset
    """
    logger.info(f"Tokenizing dataset with max_length={config.max_length}")
    
    def tokenize_function(examples):
        """Tokenize and chunk text into fixed-length sequences."""
        # Tokenize all texts
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=config.add_special_tokens,
            truncation=False,  # We'll handle chunking manually
            padding=False,  # Don't pad yet
            return_attention_mask=False  # Don't need masks during chunking
        )
        
        # Chunk into fixed-length sequences
        input_ids_list = []
        
        for input_ids in tokenized["input_ids"]:
            # Skip very short sequences
            if len(input_ids) < config.min_length:
                continue
                
            # Create sliding window chunks
            for i in range(0, len(input_ids) - config.max_length + 1, config.stride):
                chunk = input_ids[i:i + config.max_length]
                if len(chunk) == config.max_length:
                    input_ids_list.append(chunk)
        
        return {"input_ids": input_ids_list}
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.num_proc,
        remove_columns=dataset.column_names,
        batch_size=config.batch_size,
        desc="Tokenizing"
    )
    
    # Add labels (for causal language modeling, labels = input_ids)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(
        add_labels,
        batched=True,
        desc="Adding labels"
    )
    
    logger.info(f"Tokenization completed. Dataset size: {len(tokenized_dataset) if not config.streaming else 'streaming'}")
    return tokenized_dataset


def create_train_val_splits(dataset: Dataset, config: DataConfig) -> DatasetDict:
    """
    Create train/validation splits from dataset.
    
    Args:
        dataset: Input dataset
        config: Data configuration
        
    Returns:
        DatasetDict with train/validation splits
    """
    if config.streaming:
        # For streaming datasets, we can't split easily
        # Return the same dataset for both train and val
        logger.warning("Using same streaming dataset for train and validation")
        return DatasetDict({
            "train": dataset,
            "validation": dataset
        })
    
    # Shuffle dataset
    if config.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=config.shuffle_seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * config.train_split_ratio)
    val_size = total_size - train_size
    
    if val_size == 0:
        logger.warning("No validation split created (validation_split_ratio=0)")
        return DatasetDict({
            "train": dataset,
            "validation": dataset.select(range(min(100, total_size)))  # Small val set
        })
    
    # Create splits
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, total_size))
    
    logger.info(f"Created splits: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })


def validate_dataset_quality(dataset: Dataset, config: DataValidationConfig) -> Dict[str, Any]:
    """
    Validate dataset quality and content.
    
    Args:
        dataset: Dataset to validate
        config: Validation configuration
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating dataset quality")
    
    validation_results = {
        "total_samples": len(dataset) if hasattr(dataset, "__len__") else "unknown",
        "issues": [],
        "statistics": {}
    }
    
    # Sample some examples for validation
    if hasattr(dataset, "select"):
        sample_size = min(config.sample_size, len(dataset))
        sample_dataset = dataset.select(range(sample_size))
    else:
        # For streaming datasets, take first N samples
        sample_dataset = dataset.take(config.sample_size)
    
    # Validate sample content
    for i, example in enumerate(sample_dataset):
        text = example.get("text", "")
        
        # Check text length
        if len(text) < config.min_text_length:
            validation_results["issues"].append(f"Sample {i}: Text too short ({len(text)} chars)")
        elif len(text) > config.max_text_length:
            validation_results["issues"].append(f"Sample {i}: Text too long ({len(text)} chars)")
        
        # Check chemical content if enabled
        if config.check_chemical_content:
            chemical_terms = sum(1 for keyword in config.chemical_keywords if keyword.lower() in text.lower())
            if chemical_terms == 0:
                validation_results["issues"].append(f"Sample {i}: No chemical terms found")
    
    # Compute statistics
    if config.compute_statistics:
        text_lengths = [len(example.get("text", "")) for example in sample_dataset]
        validation_results["statistics"] = {
            "mean_length": sum(text_lengths) / len(text_lengths),
            "min_length": min(text_lengths),
            "max_length": max(text_lengths),
            "sample_count": len(text_lengths)
        }
    
    # Log results
    if validation_results["issues"]:
        logger.warning(f"Found {len(validation_results['issues'])} validation issues")
        for issue in validation_results["issues"][:10]:  # Log first 10 issues
            logger.warning(issue)
    else:
        logger.info("Dataset validation passed")
    
    if config.log_sample_texts:
        logger.info("Sample texts:")
        for i, example in enumerate(list(sample_dataset)[:3]):  # Log first 3 samples
            text = example.get("text", "")[:200]  # First 200 chars
            logger.info(f"Sample {i}: {text}...")
    
    return validation_results


def create_data_collator(tokenizer: PreTrainedTokenizer, config: DataConfig):
    """
    Create data collator for language modeling.
    
    Args:
        tokenizer: Tokenizer to use
        config: Data configuration
        
    Returns:
        DataCollatorForLanguageModeling instance
    """
    from transformers import DataCollatorForLanguageModeling
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8 if config.padding == "max_length" else None,
        return_tensors="pt"
    )


def estimate_memory_usage(config: DataConfig, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
    """
    Estimate memory usage for the dataset configuration.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer to use
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Rough estimates
    vocab_size = tokenizer.vocab_size
    sequence_length = config.max_length
    batch_size = getattr(config, 'batch_size', 16)
    
    # Memory per token (4 bytes for int32)
    bytes_per_token = 4
    
    # Estimate batch memory
    batch_memory_mb = (batch_size * sequence_length * bytes_per_token) / (1024 * 1024)
    
    # Estimate total dataset memory (rough)
    if config.max_samples:
        total_memory_mb = (config.max_samples * sequence_length * bytes_per_token) / (1024 * 1024)
    else:
        total_memory_mb = float('inf')  # Unknown for streaming
    
    return {
        "batch_memory_mb": batch_memory_mb,
        "total_memory_mb": total_memory_mb,
        "tokens_per_batch": batch_size * sequence_length,
        "vocab_size": vocab_size
    }
