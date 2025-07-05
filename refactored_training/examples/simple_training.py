"""
Simple training example using HuggingFace Trainer.

This example demonstrates the core Phase 1 implementation:
- HuggingFace Datasets for data loading
- Minimal custom code
- Professional experiment tracking
"""

import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import sys
import os

# Add the parent directory to path for importing advanced optimizations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from advanced_optimizations import AdvancedModelManager, create_flash_attention_model
    from enhanced_training import (
        EnhancedDataCollator, 
        EnhancedTrainer, 
        create_enhanced_training_config,
        generate_text_enhanced,
        setup_early_stopping
    )
    ADVANCED_FEATURES_AVAILABLE = True
    ENHANCED_TRAINING_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    ENHANCED_TRAINING_AVAILABLE = False
    print(f"⚠️ Advanced optimizations not available: {e}")
    print("Using standard model creation and training")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_training_config(
    output_dir: str = "./results",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    max_length: int = 256,
    **kwargs
) -> TrainingArguments:
    """
    Create training configuration with HuggingFace TrainingArguments.
    
    This replaces 100+ lines of custom training configuration with
    a simple, well-tested configuration system.
    """
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training setup
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        
        # Optimization (built-in)
        bf16=torch.cuda.is_bf16_supported(),  # Use BF16 if supported, otherwise FP32
        fp16=False,  # Disable FP16 to avoid gradient scaling issues
        gradient_accumulation_steps=4,  # Memory efficiency
        dataloader_num_workers=4,  # Parallel data loading
        
        # Evaluation and checkpointing (built-in)
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Logging (built-in)
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        report_to="none",  # Set to "wandb" for experiment tracking
        
        # Performance
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        
        **kwargs
    )


def load_and_prepare_data(
    model_name: str = "gpt2",
    dataset_name: str = "iAli61/chempile-education-dedup",
    max_length: int = 256,
    max_samples: int = 1000,  # Small for demo
    test_split_size: float = 0.1
):
    """
    Load and prepare data using HuggingFace Datasets.
    
    This replaces ~200 lines of custom data loading with
    HuggingFace's optimized data pipeline.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading dataset: {dataset_name}")
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Original dataset size: {len(dataset)}")
        
        # Limit size for demo
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples for demo")
            
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        logger.info("Falling back to a small synthetic dataset for demo")
        
        # Create a small synthetic dataset for demo
        synthetic_data = [
            "This is a chemical compound with molecular formula C6H12O6.",
            "The reaction produces water and carbon dioxide as byproducts.",
            "Catalysts are substances that increase the rate of chemical reactions.",
            "Organic chemistry deals with carbon-containing compounds.",
            "The periodic table organizes elements by their atomic properties."
        ] * (max_samples // 5)
        
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": synthetic_data})
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize text with proper padding and truncation
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # Pad to max_length for consistent sizes
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
    
    # Apply tokenization
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Add labels for causal language modeling
    def add_labels(examples):
        # For causal language modeling, labels should be the same as input_ids
        # The DataCollatorForLanguageModeling will handle the shifting during training
        # Input:  [BOS, token1, token2, token3, EOS]
        # Labels: [BOS, token1, token2, token3, EOS] -> DataCollator shifts to [token1, token2, token3, EOS, -100]
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
    
    # Create train/validation split
    split_dataset = tokenized_dataset.train_test_split(
        test_size=test_split_size,
        seed=42
    )
    
    logger.info(f"Dataset splits: train={len(split_dataset['train'])}, val={len(split_dataset['test'])}")
    
    # Show sample data before and after collation
    sample = split_dataset['train'][0]
    logger.info(f"Sample input IDs: {sample['input_ids'][:10]}...")
    logger.info(f"Sample labels (before collation): {sample['labels'][:10]}...")
    
    # Test data collator to show the shifting behavior
    temp_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    batch = temp_collator([sample])
    logger.info(f"After DataCollator - Input: {batch['input_ids'][0][:10]}...")
    logger.info(f"After DataCollator - Labels: {batch['labels'][0][:10]}...")
    
    # Show the difference to verify shifting
    input_tokens = batch['input_ids'][0][:10].tolist()
    label_tokens = batch['labels'][0][:10].tolist()
    
    logger.info("DataCollator Verification:")
    logger.info(f"  Input tokens:  {input_tokens}")
    logger.info(f"  Label tokens:  {label_tokens}")
    logger.info(f"  Labels shifted: {input_tokens[1:] == label_tokens[:-1] if len(input_tokens) > 1 else 'Too short'}")
    if input_tokens[1:] == label_tokens[:-1]:
        logger.info("  ✅ Label shifting is correct")
    else:
        logger.warning("  ❌ Label shifting is incorrect")

    return split_dataset, tokenizer


def create_model(model_name: str = "gpt2", use_flash_attention: bool = False, use_advanced_optimizations: bool = True):
    """
    Create model with HuggingFace integration and advanced optimizations.
    
    This replaces ~150 lines of custom model setup with
    HuggingFace's optimized model loading + Phase 2 optimizations.
    """
    logger.info(f"Loading model: {model_name}")
    
    if ADVANCED_FEATURES_AVAILABLE and use_advanced_optimizations:
        logger.info("🚀 Using Phase 2 advanced optimizations")
        
        # Use advanced model manager with Flash Attention support
        model, optimization_info = create_flash_attention_model(
            model_name=model_name,
            use_flash_attention=use_flash_attention,
            use_gradient_checkpointing=True,
            use_4bit=False,  # Set to True for 4-bit quantization
            use_8bit=False   # Set to True for 8-bit quantization
        )
        
        logger.info(f"⚡ Flash Attention: {optimization_info['flash_attention']}")
        logger.info(f"💾 Gradient checkpointing: {optimization_info['gradient_checkpointing']}")
        logger.info(f"🗜️ Quantization: {optimization_info['quantization']}")
        logger.info(f"📊 Memory delta: {optimization_info['memory_usage']['delta_gpu_gb']:.2f} GB")
        
    else:
        # Fallback to Phase 1 model creation
        logger.info("📦 Using Phase 1 standard model creation")
        
        model_kwargs = {
            "torch_dtype": torch.float16,  # Mixed precision
        }
        
        # Enable Flash Attention if requested and available
        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Enabled Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention not available: {e}")
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    return model


def train_chempile_model(
    model_name: str = "gpt2",
    dataset_name: str = "iAli61/chempile-education-dedup-dedup",
    output_dir: str = "./results",
    max_length: int = 256,
    batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    max_samples: int = 1000,
    use_flash_attention: bool = False,
    use_advanced_optimizations: bool = True,
    **kwargs
):
    """
    Main training function - replaces the entire TrainingPipeline class.
    
    This function demonstrates how HuggingFace Trainer replaces
    300+ lines of custom training code with ~20 lines.
    """
    logger.info("Starting ChemLLM training with HuggingFace integration")
    
    # 1. Load and prepare data (replaces custom data loading)
    dataset, tokenizer = load_and_prepare_data(
        model_name=model_name,
        dataset_name=dataset_name,
        max_length=max_length,
        max_samples=max_samples
    )
    
    # 2. Create model (replaces custom model setup)
    model = create_model(
        model_name, 
        use_flash_attention=use_flash_attention,
        use_advanced_optimizations=use_advanced_optimizations
    )
    
    # 3. Create training configuration (replaces manual config)
    if ENHANCED_TRAINING_AVAILABLE and use_advanced_optimizations:
        logger.info("Using enhanced training configuration")
        training_args = create_enhanced_training_config(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
    else:
        training_args = create_simple_training_config(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    # 4. Create data collator (replaces custom batching)
    if ENHANCED_TRAINING_AVAILABLE and use_advanced_optimizations:
        logger.info("Using enhanced data collator with label verification")
        data_collator = EnhancedDataCollator(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
    else:
        # For causal language modeling:
        # - input_ids: [token1, token2, token3, token4]
        # - labels:    [token2, token3, token4, -100]  (shifted by DataCollator)
        # The model learns to predict the next token
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM (this enables next-token prediction)
            return_tensors="pt"
        )
    
    # 5. Create trainer (replaces entire training loop)
    if ENHANCED_TRAINING_AVAILABLE and use_advanced_optimizations:
        logger.info("Using enhanced trainer with advanced features")
        trainer = EnhancedTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        # Setup early stopping if available
        trainer.add_callback(setup_early_stopping())
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    
    # 6. Train model (replaces 100+ lines of training loop)
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # 7. Save model (replaces custom checkpointing)
    trainer.save_model()
    
    # 8. Evaluate model (replaces custom evaluation)
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate()
    
    # 9. Generate sample text (demonstration)
    logger.info("Generating sample text...")
    if ENHANCED_TRAINING_AVAILABLE and use_advanced_optimizations:
        # Use enhanced text generation with better sampling
        sample_text = generate_text_enhanced(
            model=model,
            tokenizer=tokenizer,
            prompt="The chemical compound",
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        logger.info(f"Enhanced generation: {sample_text[0] if sample_text else 'No text generated'}")
    else:
        # Standard text generation
        model.eval()
        with torch.no_grad():
            prompt = "The chemical compound"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()
        
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Standard generation: {generated_text}")
    
    # Return results
    if ENHANCED_TRAINING_AVAILABLE and use_advanced_optimizations:
        generated_sample = sample_text[0] if sample_text else "No text generated"
    else:
        generated_sample = generated_text if 'generated_text' in locals() else "No text generated"
        
    results = {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "generated_sample": generated_sample,
        "model_path": training_args.output_dir
    }
    
    logger.info("Training completed successfully!")
    logger.info(f"Final results: {results}")
    
    return results


def main():
    """
    Demo function showing the simplified training approach.
    """
    print("🚀 ChemLLM Training - HuggingFace Integration Demo")
    print("=" * 60)
    
    # Simple training with minimal configuration
    results = train_chempile_model(
        model_name="gpt2",  # Small model for demo
        output_dir="./demo_results",
        max_length=128,  # Shorter sequences for demo
        batch_size=2,  # Small batch for demo
        num_epochs=1,  # Quick training
        max_samples=100,  # Very small dataset for demo
        learning_rate=5e-4,
        use_flash_attention=False  # Disable for compatibility
    )
    
    print("\n✅ Training completed!")
    print(f"📊 Train loss: {results['train_loss']:.4f}")
    print(f"📊 Eval loss: {results['eval_loss']:.4f}")
    print(f"⏱️  Training time: {results['train_runtime']:.1f}s")
    print(f"🚀 Samples/sec: {results['train_samples_per_second']:.1f}")
    print(f"📝 Generated text: {results['generated_sample']}")
    print(f"💾 Model saved to: {results['model_path']}")
    
    print("\n🎉 Demo completed! This replaced 500+ lines of custom code with ~100 lines.")


if __name__ == "__main__":
    main()
