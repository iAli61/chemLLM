#!/usr/bin/env python3
"""
Simple diagnostic script to test training components
"""

import torch
import sys
import time

print("🔧 Running Diagnostic Tests...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\n📦 Testing Imports...")
try:
    import tiktoken
    print("✅ tiktoken imported successfully")
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"✅ GPT-2 tokenizer loaded, vocab size: {tokenizer.n_vocab}")
except Exception as e:
    print(f"❌ tiktoken error: {e}")

try:
    from datasets import load_dataset
    print("✅ datasets library imported successfully")
except Exception as e:
    print(f"❌ datasets error: {e}")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    print("✅ numpy and matplotlib imported successfully")
except Exception as e:
    print(f"❌ numpy/matplotlib error: {e}")

print("\n🧪 Testing Model Components...")
try:
    # Import and test the model
    from gpt2model import GPT2
    
    config = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    print("✅ GPT2 model imported successfully")
    
    # Test model creation
    model = GPT2(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test input
    batch_size, seq_len = 2, 256
    test_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(test_input)
    forward_time = time.time() - start_time
    
    print(f"✅ Forward pass successful: {output.shape}")
    print(f"✅ Forward pass time: {forward_time:.4f}s")
    print(f"✅ Throughput: {batch_size * seq_len / forward_time:.0f} tokens/sec")
    
except Exception as e:
    print(f"❌ Model test error: {e}")
    import traceback
    traceback.print_exc()

print("\n🗂️ Testing Dataset Loading...")
try:
    DATASET_ID = "iAli61/chempile-education-dedup-dedup"
    print(f"Loading dataset: {DATASET_ID}")
    
    start_time = time.time()
    dataset = load_dataset(DATASET_ID, split="train")
    load_time = time.time() - start_time
    
    print(f"✅ Dataset loaded successfully in {load_time:.2f}s")
    print(f"✅ Dataset size: {len(dataset):,} samples")
    print(f"✅ Sample text length: {len(dataset[0]['text'])} characters")
    
except Exception as e:
    print(f"❌ Dataset loading error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Diagnostic Complete!")
print("If all tests pass ✅, your environment is ready for optimized training!")
