"""
Training with Performance Monitoring - Phase 2 Step 4

This example demonstrates comprehensive performance monitoring including:
- Weights & Biases integration  
- Real-time metrics dashboard
- Performance benchmarking
- Experiment tracking
"""

import argparse
import logging
import torch
from pathlib import Path
import sys
import os

# Add the parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

try:
    from performance_monitoring import PerformanceMonitor, TrainerWithMonitoring, create_monitoring_config
    from advanced_optimizations import AdvancedModelManager
    from enhanced_training import EnhancedTrainer, EnhancedDataCollator, create_enhanced_training_config
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Performance monitoring not available: {e}")
    MONITORING_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_monitoring_experiment(
    project_name: str = "chemllm-phase2",
    experiment_name: str = None,
    enable_wandb: bool = True,
    batch_size: int = 4,
    sequence_length: int = 256
) -> PerformanceMonitor:
    """Setup comprehensive performance monitoring."""
    if not MONITORING_AVAILABLE:
        logger.warning("📊 Performance monitoring not available - running without monitoring")
        return None
    
    monitor = PerformanceMonitor(
        project_name=project_name,
        experiment_name=experiment_name,
        enable_wandb=enable_wandb,
        log_frequency=5,  # Log every 5 steps for detailed monitoring
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    logger.info(f"🔍 Performance monitoring initialized: {experiment_name}")
    return monitor


def load_and_prepare_data(
    model_name: str = "gpt2",
    dataset_name: str = "iAli61/chempile-education-dedup-dedup", 
    max_length: int = 256,
    max_samples: int = 1000
):
    """Load and prepare dataset - same as before but with monitoring."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    logger.info(f"Original dataset size: {len(dataset)}")
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {max_samples} samples for demo")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    train_size = int(0.9 * len(tokenized_dataset))
    val_size = len(tokenized_dataset) - train_size
    
    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, train_size + val_size))
    
    dataset_dict = {
        "train": train_dataset,
        "test": val_dataset
    }
    
    logger.info(f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return dataset_dict, tokenizer


def create_model_with_monitoring(
    model_name: str,
    monitor: PerformanceMonitor = None,
    use_advanced_optimizations: bool = True,
    use_flash_attention: bool = False
):
    """Create model with performance monitoring."""
    logger.info(f"Loading model: {model_name}")
    
    if use_advanced_optimizations and MONITORING_AVAILABLE:
        logger.info("🚀 Using advanced optimizations with monitoring")
        try:
            model_manager = AdvancedModelManager()
            model, model_info = model_manager.create_optimized_model(
                model_name,
                use_flash_attention=use_flash_attention,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Log model info to monitor
            if monitor:
                config = {
                    "model_name": model_name,
                    "flash_attention": model_info.get("flash_attention", False),
                    "precision": "bf16",
                    "gradient_checkpointing": True,
                    "quantization": model_info.get("quantization", "none")
                }
                
                logger.info(f"⚡ Flash Attention: {model_info.get('flash_attention', False)}")
                logger.info(f"💾 Gradient checkpointing: True")
                logger.info(f"🗜️ Quantization: {model_info.get('quantization', 'none')}")
                logger.info(f"📊 Memory delta: {model_info.get('memory_delta_gb', 0):.2f} GB")
            
        except Exception as e:
            logger.warning(f"Advanced optimizations failed: {e}, using standard model")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model.gradient_checkpointing_enable()
    else:
        logger.info("📦 Using standard model creation")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    return model


def train_with_monitoring(
    model_name: str = "gpt2",
    dataset_name: str = "iAli61/chempile-education-dedup-dedup",
    output_dir: str = "./monitored_results",
    max_length: int = 256,
    batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    max_samples: int = 1000,
    use_flash_attention: bool = False,
    use_advanced_optimizations: bool = True,
    project_name: str = "chemllm-phase2",
    experiment_name: str = None,
    enable_wandb: bool = True,
    **kwargs
):
    """
    Main training function with comprehensive performance monitoring.
    """
    logger.info("🚀 Starting ChemLLM training with performance monitoring")
    
    # 1. Setup performance monitoring
    monitor = setup_monitoring_experiment(
        project_name=project_name,
        experiment_name=experiment_name,
        enable_wandb=enable_wandb,
        batch_size=batch_size,
        sequence_length=max_length
    )
    
    # 2. Load and prepare data
    dataset, tokenizer = load_and_prepare_data(
        model_name=model_name,
        dataset_name=dataset_name,
        max_length=max_length,
        max_samples=max_samples
    )
    
    # 3. Create model with monitoring
    model = create_model_with_monitoring(
        model_name,
        monitor=monitor,
        use_advanced_optimizations=use_advanced_optimizations,
        use_flash_attention=use_flash_attention
    )
    
    # 4. Create training configuration
    if use_advanced_optimizations and MONITORING_AVAILABLE:
        logger.info("Using enhanced training configuration with monitoring")
        training_args = create_enhanced_training_config(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=5,  # More frequent logging for monitoring
            eval_steps=25,
            save_steps=50,
            **kwargs
        )
        
        # Enhanced data collator
        data_collator = EnhancedDataCollator(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt",
            verbose_verification=True
        )
        
        # Enhanced trainer
        trainer = EnhancedTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            scheduler_type="cosine"
        )
    else:
        logger.info("Using standard training configuration")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            save_steps=50,
            bf16=True,
            **kwargs
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    
    # 5. Setup monitoring integration
    if monitor:
        # Log system and model info
        monitor.log_system_info(model, tokenizer, training_args)
        
        # Wrap trainer with monitoring
        monitored_trainer = TrainerWithMonitoring(trainer, monitor)
        
        # Create test batch for benchmarking
        test_batch = data_collator([dataset["train"][0], dataset["train"][1]])
        if torch.cuda.is_available():
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
        
        # Run initial benchmark
        benchmark_config = {
            "model_name": model_name,
            "precision": "bf16",
            "flash_attention": use_flash_attention,
            "gradient_checkpointing": True,
        }
        benchmark_results = monitor.benchmark_model(model, tokenizer, test_batch, benchmark_config)
        logger.info(f"🏃‍♂️ Initial benchmark: {benchmark_results.training_speed_samples_per_sec:.2f} samples/sec")
    else:
        monitored_trainer = trainer
    
    # 6. Train model with monitoring
    logger.info("🎯 Starting monitored training...")
    train_result = monitored_trainer.train()
    
    # 7. Save model
    monitored_trainer.save_model()
    
    # 8. Evaluate model
    logger.info("📊 Evaluating model...")
    eval_result = monitored_trainer.evaluate()
    
    # 9. Final benchmarking
    if monitor:
        # Update benchmark config with final results
        benchmark_config.update({
            "final_train_loss": train_result.training_loss,
            "final_eval_loss": eval_result["eval_loss"],
            "convergence_steps": train_result.global_step
        })
        
        final_benchmark = monitor.benchmark_model(model, tokenizer, test_batch, benchmark_config)
        logger.info(f"🏁 Final benchmark: {final_benchmark.training_speed_samples_per_sec:.2f} samples/sec")
    
    # 10. Prepare results
    results = {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "model_path": training_args.output_dir
    }
    
    if monitor:
        # Finalize monitoring
        summary = monitor.finalize()
        logger.info(f"📈 Experiment summary: {summary}")
        
        # Add monitoring stats to results
        results.update({
            "experiment_name": monitor.experiment_name,
            "dashboard_created": True,
            "monitoring_enabled": True
        })
        
        # Add final training statistics including tokens per second
        final_stats = monitor.get_final_training_stats()
        results.update(final_stats)
    
    logger.info("✅ Training with monitoring completed successfully!")
    logger.info(f"📊 Final results: {results}")
    
    return results


def main():
    """
    Demo function for training with comprehensive performance monitoring.
    """
    parser = argparse.ArgumentParser(description="ChemLLM Training with Performance Monitoring")
    parser.add_argument("--model-name", default="gpt2", help="Model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples for demo")
    parser.add_argument("--use-advanced-optimizations", action="store_true", help="Use advanced optimizations")
    parser.add_argument("--use-flash-attention", action="store_true", help="Enable Flash Attention")
    parser.add_argument("--experiment-name", help="Custom experiment name")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases")
    parser.add_argument("--project-name", default="chemllm-phase2", help="W&B project name")
    
    args = parser.parse_args()
    
    print("🚀 ChemLLM Training with Performance Monitoring - Phase 2 Step 4")
    print("=" * 70)
    print(f"📊 Experiment: {args.experiment_name or 'auto-generated'}")
    print(f"🔬 Project: {args.project_name}")
    print(f"⚡ Advanced optimizations: {args.use_advanced_optimizations}")
    print(f"🌟 Flash Attention: {args.use_flash_attention}")
    print(f"📈 W&B enabled: {not args.disable_wandb}")
    
    try:
        results = train_with_monitoring(
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            max_samples=args.max_samples,
            use_advanced_optimizations=args.use_advanced_optimizations,
            use_flash_attention=args.use_flash_attention,
            experiment_name=args.experiment_name,
            enable_wandb=not args.disable_wandb,
            project_name=args.project_name
        )
        
        print("\n🎉 Training completed successfully!")
        print("📊 Results:")
        for key, value in results.items():
            if key == "overall_tokens_per_second" and value:
                print(f"  🚀 {key}: {value:.0f}")
            elif key == "average_tokens_per_second" and value:
                print(f"  ⚡ {key}: {value:.0f}")
            else:
                print(f"  {key}: {value}")
        
        # Highlight performance metrics
        if "overall_tokens_per_second" in results or "average_tokens_per_second" in results:
            print("\n🚀 Performance Highlights:")
            if "overall_tokens_per_second" in results:
                print(f"  Overall tokens/second: {results['overall_tokens_per_second']:.0f}")
            if "average_tokens_per_second" in results:
                print(f"  Average tokens/second: {results['average_tokens_per_second']:.0f}")
            if "total_tokens_processed" in results:
                print(f"  Total tokens processed: {results['total_tokens_processed']:,}")
        
        if results.get("monitoring_enabled"):
            print("\n📈 Performance monitoring artifacts:")
            print("  - Training dashboard: ./training_dashboard.png")
            print("  - Experiment summary: ./experiment_summary.json")
            if not args.disable_wandb:
                print(f"  - W&B experiment: https://wandb.ai/{args.project_name}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
