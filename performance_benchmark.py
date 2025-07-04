#!/usr/bin/env python3
"""
Performance benchmark comparing custom MHA vs nn.MultiheadAttention
with and without torch.compile optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import os
from typing import Dict, Tuple

# Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class MHA(nn.Module):
    """Custom Multi-Head Attention implementation"""
    def __init__(self, config):
        super(MHA, self).__init__()
        # check if number of heads divides embedding dimension
        if config['emb_dim'] % config['n_heads'] != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")
        self.n_heads = config['n_heads']
        self.head_dim = config['emb_dim'] // self.n_heads
        self.config = config
        # dimensions [batch_size, sequence_length, 3 * emb_dim for q, k, v]
        self.qkv = nn.Linear(config['emb_dim'], config['emb_dim'] * 3, bias=config['qkv_bias'])
        self.attn_drop = nn.Dropout(config['drop_rate'])
        self.proj = nn.Linear(config['emb_dim'], config['emb_dim'])
        self.proj_drop = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        B, T, C = x.size()
        # B: batch size, T: sequence length, C: embedding dimension
        # (B, T, 3 * C) -> (B, T, 3, n_heads, head_dim) 
        # where n_heads = config['n_heads'] and head_dim = C // n_heads
        qkv = self.qkv(x).view(B, T, 3, self.config['n_heads'], self.head_dim)
        # (B, T, 3, n_heads, head_dim) -> (3, B, n_heads, T, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        use_dropout = 0. if not self.training else self.config['drop_rate']
        context_vec = nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=use_dropout, 
            is_causal=True
        )

        # (B, n_heads, T, head_dim) -> (B, T, n_heads * head_dim)
        context_vec = context_vec.transpose(1, 2).contiguous().view(B, T, self.config['emb_dim'])
        # (B, T, emb_dim) -> (B, T, emb_dim)
        context_vec = self.proj(context_vec)

        return context_vec

def cleanup_memory():
    """Clean up GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_memory_usage() -> str:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CPU only"

def benchmark_model(model, x, model_name: str, num_warmup: int = 3, num_runs: int = 5) -> Tuple[float, float]:
    """Benchmark a model with proper warmup and averaging"""
    model.eval()
    
    print(f"  Starting benchmark for {model_name}...")
    print(f"  {get_memory_usage()}")
    
    # Warmup runs
    for i in range(num_warmup):
        try:
            with torch.no_grad():
                if "nn.MultiheadAttention" in model_name:
                    _ = model(x, x, x)
                else:
                    _ = model(x)
            if i == 0:  # Only print for first warmup
                print(f"  Warmup {i+1}/{num_warmup} completed")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM during warmup for {model_name}")
                cleanup_memory()
                raise e
    
    # Synchronize GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Actual timing runs
    times = []
    for i in range(num_runs):
        try:
            start_time = time.perf_counter()
            with torch.no_grad():
                if "nn.MultiheadAttention" in model_name:
                    output = model(x, x, x)
                else:
                    output = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            # Clear output to save memory
            del output
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM during run {i+1} for {model_name}")
                cleanup_memory()
                raise e
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
    return avg_time, std_time

def run_benchmark():
    """Run the complete benchmark suite"""
    print("="*80)
    print("MULTI-HEAD ATTENTION PERFORMANCE BENCHMARK")
    print("="*80)
    
    # Use smaller parameters to avoid OOM
    batch_size = 64     # Reduced from 128
    seq_length = 512    # Reduced from 1024
    emb_dim = GPT_CONFIG_124M['emb_dim']
    num_warmup = 3      # Reduced from 5
    num_runs = 5        # Reduced from 10
    
    print(f"Parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Warmup runs: {num_warmup}")
    print(f"  Timing runs: {num_runs}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    cleanup_memory()
    
    # Create input data
    x = torch.randn(batch_size, seq_length, emb_dim, device=device)
    print(f"Input tensor shape: {x.shape}")
    print(f"Input tensor size: {x.numel() * 4 / 1024**2:.2f} MB")
    print()
    
    # Results storage
    results = {}
    
    try:
        # 1. Custom MHA (baseline)
        print("1. Testing Custom MHA (baseline)...")
        mha_custom = MHA(GPT_CONFIG_124M).to(device)
        avg_time, std_time = benchmark_model(mha_custom, x, "Custom MHA", num_warmup, num_runs)
        results["Custom MHA"] = {"time": avg_time, "std": std_time}
        print(f"   Result: {avg_time:.6f} ± {std_time:.6f} seconds")
        print()
        
        # 2. Custom MHA with torch.compile
        print("2. Testing Custom MHA with torch.compile...")
        mha_custom_compiled = torch.compile(mha_custom, mode="default")
        avg_time, std_time = benchmark_model(mha_custom_compiled, x, "Custom MHA Compiled", num_warmup, num_runs)
        results["Custom MHA Compiled"] = {"time": avg_time, "std": std_time}
        print(f"   Result: {avg_time:.6f} ± {std_time:.6f} seconds")
        print()
        
        # Clean up custom models
        del mha_custom, mha_custom_compiled
        cleanup_memory()
        
        # 3. nn.MultiheadAttention (baseline)
        print("3. Testing nn.MultiheadAttention (baseline)...")
        mha_nn = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=GPT_CONFIG_124M['n_heads'], 
            dropout=GPT_CONFIG_124M['drop_rate'], 
            bias=GPT_CONFIG_124M['qkv_bias'],
            batch_first=True  # Match our input format
        ).to(device)
        avg_time, std_time = benchmark_model(mha_nn, x, "nn.MultiheadAttention", num_warmup, num_runs)
        results["nn.MultiheadAttention"] = {"time": avg_time, "std": std_time}
        print(f"   Result: {avg_time:.6f} ± {std_time:.6f} seconds")
        print()
        
        # 4. nn.MultiheadAttention with torch.compile
        print("4. Testing nn.MultiheadAttention with torch.compile...")
        mha_nn_compiled = torch.compile(mha_nn, mode="default")
        avg_time, std_time = benchmark_model(mha_nn_compiled, x, "nn.MultiheadAttention Compiled", num_warmup, num_runs)
        results["nn.MultiheadAttention Compiled"] = {"time": avg_time, "std": std_time}
        print(f"   Result: {avg_time:.6f} ± {std_time:.6f} seconds")
        print()
        
        # Clean up
        del mha_nn, mha_nn_compiled, x
        cleanup_memory()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"CUDA OOM Error: {e}")
            print("Try reducing batch_size or seq_length in the script")
            return
        else:
            raise e
    
    # Print comprehensive results
    print("="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    if len(results) >= 2:
        baseline_custom = results["Custom MHA"]["time"]
        
        for name, data in results.items():
            speedup_vs_custom = baseline_custom / data["time"]
            print(f"{name:30s}: {data['time']:.6f}s ± {data['std']:.6f}s "
                  f"(speedup: {speedup_vs_custom:.2f}x)")
        
        print()
        print("Compilation Speedups:")
        if "Custom MHA Compiled" in results:
            custom_speedup = results["Custom MHA"]["time"] / results["Custom MHA Compiled"]["time"]
            print(f"Custom MHA compilation speedup: {custom_speedup:.2f}x")
        
        if "nn.MultiheadAttention Compiled" in results:
            nn_speedup = results["nn.MultiheadAttention"]["time"] / results["nn.MultiheadAttention Compiled"]["time"]
            print(f"nn.MultiheadAttention compilation speedup: {nn_speedup:.2f}x")
        
        print()
        best_performer = min(results.keys(), key=lambda k: results[k]['time'])
        print(f"Best performer: {best_performer}")
        print(f"Best time: {results[best_performer]['time']:.6f}s ± {results[best_performer]['std']:.6f}s")
    
    print("="*80)

if __name__ == "__main__":
    # Set environment variable to enable better error reporting
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    run_benchmark()
