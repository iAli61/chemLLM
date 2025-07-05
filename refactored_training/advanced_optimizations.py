"""
Advanced Model Optimizations for ChemLLM Training - Phase 2

This module provides enhanced model creation with Flash Attention, 
memory optimizations, and advanced training features.
"""

import logging
import torch
import psutil
import gc
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class AdvancedModelManager:
    """
    Advanced model manager with Flash Attention, memory optimization, 
    and performance monitoring capabilities.
    """
    
    def __init__(self):
        self.memory_stats = {}
        self.performance_metrics = {}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # System memory
        memory_info['system_ram_gb'] = psutil.virtual_memory().used / (1024**3)
        memory_info['system_ram_percent'] = psutil.virtual_memory().percent
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            memory_info['gpu_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3)
        
        return memory_info
    
    def clear_memory(self):
        """Comprehensive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def create_optimized_model(
        self,
        model_name: str = "gpt2",
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = True,
        use_8bit: bool = False,
        use_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = "auto",
        **kwargs
    ) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
        """
        Create an optimized model with advanced features.
        
        Args:
            model_name: HuggingFace model name
            use_flash_attention: Enable Flash Attention 2 if available
            use_gradient_checkpointing: Enable gradient checkpointing for memory
            use_8bit: Use 8-bit quantization (requires bitsandbytes)
            use_4bit: Use 4-bit quantization (requires bitsandbytes)
            torch_dtype: Model precision (bf16, fp16, fp32)
            device_map: Device mapping strategy
            **kwargs: Additional model arguments
        
        Returns:
            Tuple of (model, optimization_info)
        """
        logger.info(f"🤖 Creating optimized model: {model_name}")
        
        # Record initial memory
        initial_memory = self.get_memory_usage()
        
        # Determine optimal dtype
        if torch_dtype is None:
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                logger.info("📊 Using BF16 precision (optimal)")
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info("📊 Using FP16 precision")
            else:
                torch_dtype = torch.float32
                logger.info("📊 Using FP32 precision (CPU)")
        
        # Model configuration
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            **kwargs
        }
        
        # Quantization configuration
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("🗜️ Enabled 4-bit quantization")
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("🗜️ Enabled 8-bit quantization")
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Flash Attention configuration
        flash_attention_enabled = False
        if use_flash_attention:
            try:
                # Check if Flash Attention is available
                config = AutoConfig.from_pretrained(model_name)
                if hasattr(config, 'attn_implementation'):
                    model_kwargs["attn_implementation"] = "flash_attention_3"
                    flash_attention_enabled = True
                    logger.info("⚡ Flash Attention 2 enabled")
                else:
                    logger.info("⚠️ Model doesn't support Flash Attention, using default")
            except Exception as e:
                logger.warning(f"⚠️ Flash Attention setup failed: {e}, using default attention")
        
        # Load the model
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            logger.info(f"✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            # Fallback: try without Flash Attention
            if "attn_implementation" in model_kwargs:
                logger.info("🔄 Retrying without Flash Attention...")
                del model_kwargs["attn_implementation"]
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                flash_attention_enabled = False
            else:
                raise e
        
        # Enable gradient checkpointing
        if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("💾 Gradient checkpointing enabled")
        
        # Record final memory and model info
        final_memory = self.get_memory_usage()
        
        optimization_info = {
            "model_name": model_name,
            "parameters": f"{model.num_parameters():,}",
            "torch_dtype": str(torch_dtype),
            "flash_attention": flash_attention_enabled,
            "gradient_checkpointing": use_gradient_checkpointing,
            "quantization": "4bit" if use_4bit else "8bit" if use_8bit else "none",
            "memory_usage": {
                "initial": initial_memory,
                "final": final_memory,
                "delta_gpu_gb": final_memory.get('gpu_allocated_gb', 0) - initial_memory.get('gpu_allocated_gb', 0)
            }
        }
        
        logger.info(f"📊 Model info: {model.num_parameters():,} parameters")
        logger.info(f"💾 Memory delta: +{optimization_info['memory_usage']['delta_gpu_gb']:.2f} GB GPU")
        
        return model, optimization_info


def create_flash_attention_model(
    model_name: str = "gpt2",
    use_flash_attention: bool = True,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Simplified function to create a model with Flash Attention.
    
    This is a convenience wrapper around AdvancedModelManager.
    """
    manager = AdvancedModelManager()
    return manager.create_optimized_model(
        model_name=model_name,
        use_flash_attention=use_flash_attention,
        **kwargs
    )


def benchmark_attention_mechanisms(
    model_name: str = "gpt2",
    sequence_length: int = 512,
    batch_size: int = 4
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark Flash Attention vs standard attention.
    
    Returns performance metrics for both attention types.
    """
    import time
    
    logger.info("🏃 Benchmarking attention mechanisms...")
    results = {}
    
    # Test configurations
    configs = [
        {"name": "standard", "use_flash_attention": False},
        {"name": "flash_attention_3", "use_flash_attention": True}
    ]
    
    manager = AdvancedModelManager()
    
    for config in configs:
        logger.info(f"Testing {config['name']}...")
        
        try:
            # Clear memory before each test
            manager.clear_memory()
            initial_memory = manager.get_memory_usage()
            
            # Create model
            start_time = time.time()
            model, model_info = manager.create_optimized_model(
                model_name=model_name,
                use_flash_attention=config["use_flash_attention"]
            )
            load_time = time.time() - start_time
            
            # Create dummy input
            dummy_input = torch.randint(0, 1000, (batch_size, sequence_length))
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                model = model.cuda()
            
            # Warmup
            with torch.no_grad():
                _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark forward pass
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):  # Average over 5 runs
                    _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            forward_time = (time.time() - start_time) / 5
            final_memory = manager.get_memory_usage()
            
            results[config['name']] = {
                "load_time_seconds": load_time,
                "forward_time_seconds": forward_time,
                "memory_gb": final_memory.get('gpu_allocated_gb', 0),
                "flash_attention_enabled": model_info["flash_attention"],
                "parameters": model_info["parameters"]
            }
            
            # Cleanup
            del model
            manager.clear_memory()
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed for {config['name']}: {e}")
            results[config['name']] = {"error": str(e)}
    
    # Calculate improvements
    if "standard" in results and "flash_attention_3" in results:
        std_time = results["standard"].get("forward_time_seconds", float('inf'))
        fa_time = results["flash_attention_3"].get("forward_time_seconds", float('inf'))
        
        if std_time > 0 and fa_time > 0:
            speedup = std_time / fa_time
            results["performance_improvement"] = {
                "speedup_ratio": speedup,
                "time_reduction_percent": ((std_time - fa_time) / std_time) * 100
            }
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Advanced Model Manager")
    print("=" * 50)
    
    # Create optimized model
    model, info = create_flash_attention_model("gpt2", use_flash_attention=True)
    
    print(f"✅ Model created: {info['parameters']} parameters")
    print(f"⚡ Flash Attention: {info['flash_attention']}")
    print(f"💾 Memory usage: {info['memory_usage']['delta_gpu_gb']:.2f} GB")
    
    # Run benchmark
    print("\n🏃 Running attention benchmark...")
    benchmark_results = benchmark_attention_mechanisms("gpt2")
    
    for method, metrics in benchmark_results.items():
        if "error" not in metrics:
            print(f"\n{method}:")
            print(f"  Forward time: {metrics.get('forward_time_seconds', 0):.4f}s")
            print(f"  Memory: {metrics.get('memory_gb', 0):.2f} GB")
