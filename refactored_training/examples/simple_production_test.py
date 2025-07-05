"""
Simple Production Features Demo
==============================

Minimal demonstration of production features without complex dependencies.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_versioning():
    """Test basic model versioning functionality"""
    logger.info("📦 Testing Model Versioning")
    
    try:
        from production_features import ModelVersionManager, ProductionConfig
        
        # Create config
        config = ProductionConfig(
            enable_versioning=True,
            model_registry_dir="./test_registry"
        )
        
        # Create version manager
        version_manager = ModelVersionManager(config)
        
        # Test version generation
        v1 = version_manager.get_next_version("patch")
        logger.info(f"Next patch version: {v1}")
        
        v2 = version_manager.get_next_version("minor")  
        logger.info(f"Next minor version: {v2}")
        
        # Create mock model directory
        os.makedirs("./mock_model", exist_ok=True)
        with open("./mock_model/config.json", 'w') as f:
            json.dump({"model_type": "gpt2"}, f)
        
        # Register mock model
        version = version_manager.register_model(
            model_path="./mock_model",
            metrics={"eval_loss": 2.5, "perplexity": 12.2},
            hyperparameters={"learning_rate": 2e-5, "batch_size": 8},
            notes="Test model"
        )
        
        logger.info(f"✅ Model registered with version: {version}")
        
        # List models
        models = version_manager.list_models()
        logger.info(f"📋 Registered models: {len(models)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model versioning test failed: {e}")
        return False


def test_hyperparameter_optimizer():
    """Test hyperparameter optimizer without running full optimization"""
    logger.info("🔍 Testing Hyperparameter Optimizer")
    
    try:
        from production_features import HyperparameterOptimizer, ProductionConfig
        import optuna
        
        # Create config
        config = ProductionConfig(
            enable_hpo=True,
            hpo_trials=2,
            hpo_study_name="test_study"
        )
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(config)
        
        # Test study creation
        study = optimizer.create_study()
        logger.info(f"✅ Study created: {study.study_name}")
        
        # Test hyperparameter suggestion
        trial = optuna.create_trial(
            params={"learning_rate": 2e-5, "batch_size": 4},
            value=2.5
        )
        
        suggested_params = optimizer.suggest_hyperparameters(trial)
        logger.info(f"📊 Sample hyperparameters: {list(suggested_params.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter optimizer test failed: {e}")
        return False


def test_production_config():
    """Test production configuration"""
    logger.info("⚙️ Testing Production Configuration")
    
    try:
        from production_features import ProductionConfig
        
        # Test default config
        config = ProductionConfig()
        logger.info(f"✅ Default config: HPO={config.enable_hpo}, Versioning={config.enable_versioning}")
        
        # Test custom config
        custom_config = ProductionConfig(
            enable_hpo=True,
            hpo_trials=10,
            enable_versioning=True,
            wandb_project="test-project"
        )
        logger.info(f"✅ Custom config: Project={custom_config.wandb_project}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Production config test failed: {e}")
        return False


def main():
    """Run simple tests"""
    print("🚀 ChemLLM Production Features - Simple Test")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_production_config),
        ("Model Versioning", test_model_versioning),
        ("Hyperparameter Optimizer", test_hyperparameter_optimizer)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🧪 Running {test_name} test...")
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Results:")
    print("=" * 30)
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Production features are working.")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
