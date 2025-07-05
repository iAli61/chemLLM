"""
Production-Ready Features for ChemLLM Training
==============================================

This module provides production-ready features including:
- Hyperparameter optimization with Optuna
- Model versioning and experiment tracking
- Automated model selection and validation
- Deployment-ready model packaging
- Advanced experiment management

Author: Assistant
Date: 2025-01-05
"""

import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import logging
from dataclasses import dataclass, asdict
import optuna
# from optuna.integration import PyTorchLightningPruningCallback  # Not needed for basic optuna
try:
    import wandb
except ImportError:
    wandb = None
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
    AutoModelForCausalLM
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production features"""
    
    # Hyperparameter optimization
    enable_hpo: bool = False
    hpo_trials: int = 20
    hpo_timeout: Optional[int] = None  # seconds
    hpo_study_name: str = "chemllm_optimization"
    hpo_storage: Optional[str] = None  # SQLite database path
    
    # Model versioning
    enable_versioning: bool = True
    version_format: str = "v{major}.{minor}.{patch}"
    auto_version_bump: bool = True
    
    # Experiment tracking
    experiment_dir: str = "./experiments"
    model_registry_dir: str = "./model_registry"
    enable_wandb: bool = True
    wandb_project: str = "chemllm-production"
    
    # Model validation
    validation_metrics: List[str] = None
    min_performance_threshold: Optional[float] = None
    
    # Deployment
    export_formats: List[str] = None  # ["onnx", "torchscript", "huggingface"]
    deployment_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.validation_metrics is None:
            self.validation_metrics = ["eval_loss", "perplexity"]
        if self.export_formats is None:
            self.export_formats = ["huggingface"]
        if self.deployment_config is None:
            self.deployment_config = {
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.study = None
        self.best_trial = None
        
    def create_study(self) -> optuna.Study:
        """Create Optuna study for hyperparameter optimization"""
        storage = self.config.hpo_storage
        if storage and not storage.startswith("sqlite"):
            storage = f"sqlite:///{storage}"
            
        self.study = optuna.create_study(
            study_name=self.config.hpo_study_name,
            direction="minimize",  # Minimize validation loss
            storage=storage,
            load_if_exists=True
        )
        
        logger.info(f"🔍 Created Optuna study: {self.config.hpo_study_name}")
        return self.study
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""
        return {
            # Learning rate optimization
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            
            # Training dynamics
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [1, 2, 4, 8]
            ),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [1, 2, 4, 8]
            ),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
            
            # Model architecture (if applicable)
            "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3),
            "hidden_dropout": trial.suggest_float("hidden_dropout", 0.0, 0.3),
            
            # Advanced optimizations
            "dataloader_num_workers": trial.suggest_int("dataloader_num_workers", 0, 8),
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"]
            )
        }
    
    def optimize(
        self, 
        objective_func,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> optuna.Trial:
        """Run hyperparameter optimization"""
        if self.study is None:
            self.create_study()
            
        n_trials = n_trials or self.config.hpo_trials
        timeout = timeout or self.config.hpo_timeout
        
        logger.info(f"🚀 Starting hyperparameter optimization: {n_trials} trials")
        
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._log_trial_callback]
        )
        
        self.best_trial = self.study.best_trial
        logger.info(f"✅ Optimization completed. Best value: {self.best_trial.value:.4f}")
        logger.info(f"📊 Best parameters: {self.best_trial.params}")
        
        return self.best_trial
    
    def _log_trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback to log trial progress"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(
                f"⚡ Trial {trial.number}: {trial.value:.4f} "
                f"(best: {study.best_value:.4f})"
            )


class ModelVersionManager:
    """Model versioning and registry management"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.registry_dir = Path(config.model_registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize version tracking
        self.version_file = self.registry_dir / "versions.json"
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict[str, Any]:
        """Load version history"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {"current_version": "v0.0.0", "history": []}
    
    def _save_versions(self):
        """Save version history"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def get_next_version(self, bump_type: str = "patch") -> str:
        """Get next version number"""
        current = self.versions["current_version"]
        
        # Parse current version
        if current.startswith('v'):
            current = current[1:]
        
        parts = current.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Bump version
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
            
        return f"v{major}.{minor}.{patch}"
    
    def register_model(
        self,
        model_path: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        version: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """Register a new model version"""
        if version is None:
            version = self.get_next_version()
            
        timestamp = datetime.now().isoformat()
        
        # Create version directory
        version_dir = self.registry_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # Copy model files
        if os.path.isdir(model_path):
            shutil.copytree(model_path, version_dir / "model", dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, version_dir / "model")
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": timestamp,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "notes": notes,
            "model_path": str(version_dir / "model")
        }
        
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update version history
        self.versions["current_version"] = version
        self.versions["history"].append(metadata)
        self._save_versions()
        
        logger.info(f"📦 Model registered: {version}")
        logger.info(f"📊 Metrics: {metrics}")
        
        return version
    
    def get_best_model(self, metric: str = "eval_loss", minimize: bool = True) -> Dict[str, Any]:
        """Get the best model based on a metric"""
        if not self.versions["history"]:
            raise ValueError("No models registered")
        
        history = self.versions["history"]
        
        # Filter models that have the metric
        valid_models = [m for m in history if metric in m.get("metrics", {})]
        
        if not valid_models:
            raise ValueError(f"No models found with metric: {metric}")
        
        # Find best model
        key_func = lambda m: m["metrics"][metric]
        best_model = min(valid_models, key=key_func) if minimize else max(valid_models, key=key_func)
        
        logger.info(f"🏆 Best model: {best_model['version']} ({metric}: {best_model['metrics'][metric]:.4f})")
        
        return best_model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return self.versions["history"]


class ExperimentManager:
    """Advanced experiment management and tracking"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.experiment_dir = Path(config.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B if enabled
        self.wandb_enabled = config.enable_wandb and wandb is not None
        if self.wandb_enabled:
            self.wandb = wandb
        elif config.enable_wandb and wandb is None:
            logger.warning("W&B not installed, disabling W&B tracking")
            self.wandb_enabled = False
    
    def start_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """Start a new experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save experiment config
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize W&B
        if self.wandb_enabled:
            self.wandb.init(
                project=self.config.wandb_project,
                name=experiment_id,
                config=config,
                tags=tags,
                dir=str(exp_dir)
            )
        
        logger.info(f"🚀 Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to tracking systems"""
        if self.wandb_enabled:
            self.wandb.log(metrics, step=step)
    
    def finish_experiment(self, final_metrics: Dict[str, float]):
        """Finish current experiment"""
        if self.wandb_enabled:
            self.wandb.log(final_metrics)
            self.wandb.finish()
        
        logger.info(f"✅ Experiment completed with metrics: {final_metrics}")


class ProductionTrainingPipeline:
    """Complete production training pipeline with all features"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.hpo = HyperparameterOptimizer(config) if config.enable_hpo else None
        self.version_manager = ModelVersionManager(config) if config.enable_versioning else None
        self.experiment_manager = ExperimentManager(config)
        
    def train_with_optimization(
        self,
        train_func,
        base_config: Dict[str, Any],
        experiment_name: str = "production_training"
    ) -> Dict[str, Any]:
        """Train model with hyperparameter optimization"""
        
        if self.config.enable_hpo and self.hpo:
            logger.info("🔍 Starting hyperparameter optimization...")
            
            def objective(trial):
                # Get suggested hyperparameters
                suggested_params = self.hpo.suggest_hyperparameters(trial)
                
                # Merge with base config
                trial_config = {**base_config, **suggested_params}
                
                # Start experiment
                exp_id = self.experiment_manager.start_experiment(
                    f"{experiment_name}_trial_{trial.number}",
                    trial_config,
                    tags=["hpo", "trial"]
                )
                
                try:
                    # Train model
                    results = train_func(trial_config)
                    
                    # Extract validation loss
                    val_loss = results.get("eval_loss", float('inf'))
                    
                    # Log to experiment manager
                    self.experiment_manager.log_metrics({"val_loss": val_loss})
                    self.experiment_manager.finish_experiment({"final_val_loss": val_loss})
                    
                    return val_loss
                    
                except Exception as e:
                    logger.error(f"❌ Trial {trial.number} failed: {e}")
                    self.experiment_manager.finish_experiment({"error": str(e)})
                    raise optuna.exceptions.TrialPruned()
            
            # Run optimization
            best_trial = self.hpo.optimize(objective)
            best_config = {**base_config, **best_trial.params}
            
            logger.info("🏆 Using best hyperparameters for final training...")
            
        else:
            best_config = base_config
        
        # Final training with best config
        exp_id = self.experiment_manager.start_experiment(
            f"{experiment_name}_final",
            best_config,
            tags=["final", "production"]
        )
        
        final_results = train_func(best_config)
        
        # Register model if versioning is enabled
        if self.config.enable_versioning and self.version_manager:
            model_path = final_results.get("model_path", "./model")
            metrics = {k: v for k, v in final_results.items() if isinstance(v, (int, float))}
            
            version = self.version_manager.register_model(
                model_path=model_path,
                metrics=metrics,
                hyperparameters=best_config,
                notes=f"Production training - {experiment_name}"
            )
            
            final_results["model_version"] = version
        
        # Log final metrics
        final_metrics = {k: v for k, v in final_results.items() if isinstance(v, (int, float))}
        self.experiment_manager.finish_experiment(final_metrics)
        
        return final_results
    
    def validate_model(self, model_path: str, validation_func) -> Dict[str, float]:
        """Validate model performance"""
        logger.info("🔍 Validating model performance...")
        
        metrics = validation_func(model_path)
        
        # Check minimum performance threshold
        if self.config.min_performance_threshold:
            primary_metric = metrics.get("eval_loss", float('inf'))
            if primary_metric > self.config.min_performance_threshold:
                logger.warning(
                    f"⚠️ Model performance below threshold: "
                    f"{primary_metric:.4f} > {self.config.min_performance_threshold:.4f}"
                )
        
        logger.info(f"✅ Model validation completed: {metrics}")
        return metrics
    
    def export_model(self, model_path: str, output_dir: str) -> Dict[str, str]:
        """Export model in various formats for deployment"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_models = {}
        
        for format_type in self.config.export_formats:
            if format_type == "huggingface":
                # Copy HuggingFace model
                hf_dir = output_dir / "huggingface"
                if os.path.isdir(model_path):
                    shutil.copytree(model_path, hf_dir, dirs_exist_ok=True)
                else:
                    hf_dir.mkdir(exist_ok=True)
                    shutil.copy2(model_path, hf_dir)
                
                # Add deployment config
                with open(hf_dir / "deployment_config.json", 'w') as f:
                    json.dump(self.config.deployment_config, f, indent=2)
                
                exported_models["huggingface"] = str(hf_dir)
                
            elif format_type == "onnx":
                # TODO: Implement ONNX export
                logger.warning("ONNX export not yet implemented")
                
            elif format_type == "torchscript":
                # TODO: Implement TorchScript export
                logger.warning("TorchScript export not yet implemented")
        
        logger.info(f"📦 Model exported in formats: {list(exported_models.keys())}")
        return exported_models


def create_production_pipeline(
    enable_hpo: bool = True,
    hpo_trials: int = 10,
    enable_versioning: bool = True,
    enable_wandb: bool = True,
    **kwargs
) -> ProductionTrainingPipeline:
    """Create a production training pipeline with specified features"""
    
    config = ProductionConfig(
        enable_hpo=enable_hpo,
        hpo_trials=hpo_trials,
        enable_versioning=enable_versioning,
        enable_wandb=enable_wandb,
        **kwargs
    )
    
    return ProductionTrainingPipeline(config)


# Example usage functions
def example_hyperparameter_optimization():
    """Example of hyperparameter optimization"""
    
    def training_function(config):
        """Mock training function"""
        # Simulate training with random performance
        import random
        eval_loss = random.uniform(2.0, 4.0)
        return {"eval_loss": eval_loss, "model_path": "./mock_model"}
    
    # Create production pipeline
    pipeline = create_production_pipeline(
        enable_hpo=True,
        hpo_trials=5,
        enable_versioning=True,
        enable_wandb=False  # Disable for example
    )
    
    # Base configuration
    base_config = {
        "num_train_epochs": 1,
        "eval_steps": 50,
        "save_steps": 100
    }
    
    # Run training with optimization
    results = pipeline.train_with_optimization(
        train_func=training_function,
        base_config=base_config,
        experiment_name="example_optimization"
    )
    
    logger.info(f"🎉 Training completed: {results}")
    return results


if __name__ == "__main__":
    # Run example
    example_hyperparameter_optimization()
