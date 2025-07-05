"""
Production Training Script with Full Feature Integration
======================================================

This script demonstrates the complete production training pipeline with:
- Hyperparameter optimization
- Model versioning
- Advanced optimizations
- Performance monitoring
- Production-ready features

Author: Assistant
Date: 2025-01-05
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from production_features import ProductionTrainingPipeline, ProductionConfig
from performance_monitoring import PerformanceMonitor, TrainerWithMonitoring
from advanced_optimizations import AdvancedModelManager
from enhanced_training import EnhancedTrainer, EnhancedDataCollator, create_enhanced_training_args

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTrainingEngine:
    """Complete production training engine"""
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.performance_monitor = None
        self.model_manager = None
        
    def setup_tokenizer_and_model(self):
        """Setup tokenizer and model with advanced optimizations"""
        logger.info(f"🤖 Loading tokenizer and model: {self.args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup advanced model manager
        self.model_manager = AdvancedModelManager(
            use_flash_attention=self.args.use_flash_attention,
            quantization_type=self.args.quantization if self.args.quantization != "none" else None,
            precision_type=self.args.precision,
            enable_gradient_checkpointing=self.args.gradient_checkpointing,
            max_memory_mb=self.args.max_memory
        )
        
        # Load model with optimizations
        self.model = self.model_manager.load_model(
            model_name=self.args.model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info(f"✅ Model loaded with {self.model.num_parameters():,} parameters")
        
    def setup_datasets(self):
        """Setup training and evaluation datasets"""
        logger.info(f"📊 Loading dataset: {self.args.dataset_name}")
        
        # Load dataset
        dataset = load_dataset(self.args.dataset_name, split="train")
        
        # Limit samples if specified
        if self.args.max_samples:
            dataset = dataset.select(range(min(self.args.max_samples, len(dataset))))
            logger.info(f"📊 Limited to {len(dataset)} samples")
        
        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.args.max_length,
                return_overflowing_tokens=False
            )
        
        # Tokenize dataset
        logger.info("🔤 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        # Split into train/eval
        split_dataset = tokenized_dataset.train_test_split(
            test_size=self.args.eval_split,
            seed=42
        )
        
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]
        
        logger.info(f"📊 Dataset splits: train={len(self.train_dataset)}, eval={len(self.eval_dataset)}")
        
    def setup_performance_monitoring(self):
        """Setup performance monitoring"""
        if self.args.enable_monitoring:
            self.performance_monitor = PerformanceMonitor(
                experiment_name=self.args.experiment_name,
                project_name="chemllm-production",
                enable_wandb=self.args.enable_wandb,
                log_interval=self.args.log_interval
            )
            
            # Log system info
            self.performance_monitor.log_system_info(
                model_params=self.model.num_parameters(),
                gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            )
            
            logger.info("📊 Performance monitoring enabled")
        
    def create_training_arguments(self, hyperparams: Optional[Dict[str, Any]] = None) -> TrainingArguments:
        """Create training arguments with optional hyperparameters"""
        
        # Base arguments
        base_args = {
            "output_dir": self.args.output_dir,
            "num_train_epochs": self.args.num_epochs,
            "per_device_train_batch_size": self.args.batch_size,
            "per_device_eval_batch_size": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "warmup_ratio": self.args.warmup_ratio,
            "max_grad_norm": self.args.max_grad_norm,
            "lr_scheduler_type": self.args.lr_scheduler_type,
            "logging_steps": self.args.log_interval,
            "eval_steps": self.args.eval_steps,
            "save_steps": self.args.save_steps,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "wandb" if self.args.enable_wandb else "none",
            "run_name": self.args.experiment_name,
            "dataloader_num_workers": self.args.num_workers,
            "fp16": self.args.precision == "fp16",
            "bf16": self.args.precision == "bf16",
            "gradient_checkpointing": self.args.gradient_checkpointing,
            "remove_unused_columns": False,
            "seed": 42
        }
        
        # Override with hyperparameters if provided
        if hyperparams:
            base_args.update(hyperparams)
            logger.info(f"🔧 Using hyperparameters: {hyperparams}")
        
        return TrainingArguments(**base_args)
    
    def train_model_function(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Training function for use with production pipeline"""
        
        # Create training arguments
        training_args = self.create_training_arguments(config)
        
        # Create data collator
        if self.args.use_enhanced_training:
            data_collator = EnhancedDataCollator(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8 if self.args.precision in ["fp16", "bf16"] else None
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8 if self.args.precision in ["fp16", "bf16"] else None
            )
        
        # Create trainer
        if self.args.use_enhanced_training:
            trainer = EnhancedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
        else:
            from transformers import Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
        
        # Wrap with monitoring if enabled
        if self.performance_monitor:
            trainer = TrainerWithMonitoring(
                trainer=trainer,
                monitor=self.performance_monitor,
                batch_size=training_args.per_device_train_batch_size,
                sequence_length=self.args.max_length
            )
        
        # Train model
        logger.info("🚀 Starting training...")
        train_result = trainer.train()
        
        # Evaluate model
        logger.info("📊 Evaluating model...")
        eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model()
        
        # Compile results
        results = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "train_runtime": train_result.metrics["train_runtime"],
            "model_path": training_args.output_dir,
            "perplexity": torch.exp(torch.tensor(eval_result["eval_loss"])).item()
        }
        
        # Add performance monitoring results if available
        if self.performance_monitor:
            monitoring_results = self.performance_monitor.get_experiment_summary()
            results.update(monitoring_results)
        
        return results
    
    def run_production_training(self):
        """Run complete production training pipeline"""
        
        # Create production configuration
        production_config = ProductionConfig(
            enable_hpo=self.args.enable_hpo,
            hpo_trials=self.args.hpo_trials,
            hpo_timeout=self.args.hpo_timeout,
            hpo_study_name=f"{self.args.experiment_name}_hpo",
            enable_versioning=self.args.enable_versioning,
            experiment_dir=self.args.experiments_dir,
            model_registry_dir=self.args.model_registry_dir,
            enable_wandb=self.args.enable_wandb,
            wandb_project="chemllm-production",
            validation_metrics=["eval_loss", "perplexity"],
            export_formats=["huggingface"]
        )
        
        # Create production pipeline
        pipeline = ProductionTrainingPipeline(production_config)
        
        # Base configuration for training
        base_config = {
            "num_train_epochs": self.args.num_epochs,
            "eval_steps": self.args.eval_steps,
            "save_steps": self.args.save_steps
        }
        
        # Run training with full production features
        logger.info("🏭 Starting production training pipeline...")
        
        results = pipeline.train_with_optimization(
            train_func=self.train_model_function,
            base_config=base_config,
            experiment_name=self.args.experiment_name
        )
        
        # Export model for deployment if requested
        if self.args.export_model:
            export_dir = f"{self.args.output_dir}_deployment"
            exported_models = pipeline.export_model(
                model_path=results["model_path"],
                output_dir=export_dir
            )
            results["exported_models"] = exported_models
        
        return results
    
    def run(self):
        """Run the complete production training process"""
        try:
            logger.info("🚀 Starting ChemLLM Production Training")
            logger.info("=" * 60)
            
            # Setup components
            self.setup_tokenizer_and_model()
            self.setup_datasets()
            self.setup_performance_monitoring()
            
            # Run production training
            results = self.run_production_training()
            
            # Display results
            logger.info("🎉 Production training completed successfully!")
            logger.info("📊 Final Results:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ChemLLM Production Training")
    
    # Model and data
    parser.add_argument("--model-name", default="gpt2", help="Model name")
    parser.add_argument("--dataset-name", default="iAli61/chempile-education-dedup-dedup", help="Dataset name")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Evaluation split ratio")
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--lr-scheduler-type", default="cosine", help="LR scheduler type")
    
    # Advanced optimizations
    parser.add_argument("--use-flash-attention", action="store_true", help="Use Flash Attention")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none", help="Quantization type")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16", help="Precision type")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--max-memory", type=int, help="Max memory in MB")
    
    # Enhanced training
    parser.add_argument("--use-enhanced-training", action="store_true", help="Use enhanced training features")
    
    # Production features
    parser.add_argument("--enable-hpo", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--hpo-trials", type=int, default=10, help="Number of HPO trials")
    parser.add_argument("--hpo-timeout", type=int, help="HPO timeout in seconds")
    parser.add_argument("--enable-versioning", action="store_true", help="Enable model versioning")
    parser.add_argument("--export-model", action="store_true", help="Export model for deployment")
    
    # Monitoring and logging
    parser.add_argument("--enable-monitoring", action="store_true", help="Enable performance monitoring")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save steps")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
    # Output directories
    parser.add_argument("--output-dir", default="./production_model", help="Output directory")
    parser.add_argument("--experiments-dir", default="./experiments", help="Experiments directory")
    parser.add_argument("--model-registry-dir", default="./model_registry", help="Model registry directory")
    parser.add_argument("--experiment-name", default="chemllm_production", help="Experiment name")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create and run training engine
    engine = ProductionTrainingEngine(args)
    results = engine.run()
    
    return results


if __name__ == "__main__":
    main()
