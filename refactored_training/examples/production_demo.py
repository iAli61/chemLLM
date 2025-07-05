"""
Production Features Demo
=======================

Simple demonstration of production features including:
- Hyperparameter optimization
- Model versioning
- Basic production pipeline

Author: Assistant
Date: 2025-01-05
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production_features import create_production_pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_training_function(config):
    """Mock training function for demonstration"""
    import random
    import time
    import json
    import os
    
    logger.info(f"🚀 Training with config: {config}")
    
    # Simulate training time
    time.sleep(1)
    
    # Simulate performance based on hyperparameters
    lr = config.get("learning_rate", 5e-5)
    batch_size = config.get("per_device_train_batch_size", 4)
    
    # Better performance with certain hyperparameter ranges
    base_loss = 3.5
    lr_penalty = abs(lr - 2e-5) * 1000  # Optimal around 2e-5
    batch_penalty = abs(batch_size - 8) * 0.1  # Optimal around 8
    
    eval_loss = base_loss + lr_penalty + batch_penalty + random.uniform(-0.2, 0.2)
    eval_loss = max(eval_loss, 2.0)  # Minimum loss
    
    # Create a mock model file for demonstration
    model_path = "./demo_model"
    os.makedirs(model_path, exist_ok=True)
    
    # Create mock model files
    with open(f"{model_path}/config.json", 'w') as f:
        json.dump({"model_type": "gpt2", "vocab_size": 50257}, f)
    
    with open(f"{model_path}/pytorch_model.bin", 'w') as f:
        f.write("mock model weights")
    
    return {
        "eval_loss": eval_loss,
        "perplexity": 2.71 ** eval_loss,
        "train_loss": eval_loss + 0.1,
        "model_path": model_path
    }


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization"""
    logger.info("🔍 Demo: Hyperparameter Optimization")
    logger.info("=" * 50)
    
    # Create production pipeline with HPO enabled
    pipeline = create_production_pipeline(
        enable_hpo=True,
        hpo_trials=5,  # Small number for demo
        enable_versioning=True,
        enable_wandb=False,  # Disable for demo
        experiment_dir="./demo_experiments",
        model_registry_dir="./demo_registry"
    )
    
    # Base configuration
    base_config = {
        "num_train_epochs": 1,
        "eval_steps": 50,
        "save_steps": 100
    }
    
    # Run training with optimization
    results = pipeline.train_with_optimization(
        train_func=mock_training_function,
        base_config=base_config,
        experiment_name="hpo_demo"
    )
    
    logger.info("🎉 HPO Demo completed!")
    logger.info(f"📊 Best results: {results}")
    
    return results


def demo_model_versioning():
    """Demonstrate model versioning"""
    logger.info("📦 Demo: Model Versioning")
    logger.info("=" * 50)
    
    import json
    import os
    
    # Create production pipeline
    pipeline = create_production_pipeline(
        enable_hpo=False,
        enable_versioning=True,
        enable_wandb=False,
        model_registry_dir="./demo_registry"
    )
    
    # Register a few mock models
    version_manager = pipeline.version_manager
    
    # Create mock model directories
    for i in range(1, 4):
        model_path = f"./mock_model_{i}"
        os.makedirs(model_path, exist_ok=True)
        
        # Create mock model files
        with open(f"{model_path}/config.json", 'w') as f:
            json.dump({"model_type": "gpt2", "vocab_size": 50257}, f)
        
        with open(f"{model_path}/pytorch_model.bin", 'w') as f:
            f.write(f"mock model weights {i}")
    
    # Register model 1
    version1 = version_manager.register_model(
        model_path="./mock_model_1",
        metrics={"eval_loss": 3.2, "perplexity": 24.5},
        hyperparameters={"learning_rate": 5e-5, "batch_size": 4},
        notes="Initial model"
    )
    
    # Register model 2 (better performance)
    version2 = version_manager.register_model(
        model_path="./mock_model_2",
        metrics={"eval_loss": 2.8, "perplexity": 16.4},
        hyperparameters={"learning_rate": 2e-5, "batch_size": 8},
        notes="Improved hyperparameters"
    )
    
    # Register model 3 (worse performance)
    version3 = version_manager.register_model(
        model_path="./mock_model_3",
        metrics={"eval_loss": 3.5, "perplexity": 30.0},
        hyperparameters={"learning_rate": 1e-4, "batch_size": 2},
        notes="High learning rate experiment"
    )
    
    # List all models
    logger.info("📋 All registered models:")
    for model in version_manager.list_models():
        logger.info(f"  {model['version']}: loss={model['metrics']['eval_loss']:.2f}")
    
    # Get best model
    best_model = version_manager.get_best_model("eval_loss")
    logger.info(f"🏆 Best model: {best_model['version']} (loss: {best_model['metrics']['eval_loss']:.2f})")
    
    return version_manager


def demo_complete_pipeline():
    """Demonstrate complete production pipeline"""
    logger.info("🏭 Demo: Complete Production Pipeline")
    logger.info("=" * 50)
    
    # Create production pipeline with all features
    pipeline = create_production_pipeline(
        enable_hpo=True,
        hpo_trials=3,  # Small number for demo
        enable_versioning=True,
        enable_wandb=False,
        experiment_dir="./demo_experiments",
        model_registry_dir="./demo_registry"
    )
    
    # Base configuration
    base_config = {
        "num_train_epochs": 1,
        "eval_steps": 50,
        "save_steps": 100
    }
    
    # Run complete pipeline
    results = pipeline.train_with_optimization(
        train_func=mock_training_function,
        base_config=base_config,
        experiment_name="complete_demo"
    )
    
    logger.info("🎉 Complete pipeline demo completed!")
    logger.info(f"📊 Final results: {results}")
    
    # Show model registry
    logger.info("📦 Model registry after training:")
    for model in pipeline.version_manager.list_models():
        logger.info(f"  {model['version']}: loss={model['metrics']['eval_loss']:.2f}")
    
    return results


def main():
    """Run all demos"""
    print("🚀 ChemLLM Production Features Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Hyperparameter optimization
        demo_hyperparameter_optimization()
        print("\n")
        
        # Demo 2: Model versioning
        demo_model_versioning()
        print("\n")
        
        # Demo 3: Complete pipeline
        demo_complete_pipeline()
        
        print("\n🎉 All demos completed successfully!")
        print("📁 Check the following directories for artifacts:")
        print("  - ./demo_experiments/ - Experiment tracking")
        print("  - ./demo_registry/ - Model registry")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
