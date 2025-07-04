# Batch Size and Memory Optimization Strategies

import torch
import math

def find_optimal_batch_size(model, tokenizer, sample_data, device, max_length=256):
    """
    Automatically find the optimal batch size for your hardware
    """
    model.eval()
    
    # Start with a reasonable batch size
    batch_size = 2  # Start smaller
    max_batch_size = 32  # Lower max
    
    # Create sample batch
    sample_input = torch.randint(0, 50257, (1, max_length), device=device)
    
    print("🔍 Finding optimal batch size...")
    
    # Clear initial memory
    torch.cuda.empty_cache()
    
    while batch_size <= max_batch_size:
        try:
            # Test forward pass
            test_input = sample_input.repeat(batch_size, 1)
            
            # Clear memory before test
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                _ = model(test_input)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_percent = (memory_used / memory_total) * 100
            
            print(f"   Batch size {batch_size}: {memory_percent:.1f}% GPU memory")
            
            # Be more conservative - stop at 60% instead of 85%
            if memory_percent > 60:  
                optimal_batch_size = max(batch_size // 2, 1)
                break
                
            # Test a forward+backward pass for more realistic memory usage
            test_input.requires_grad_(True)
            try:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    output = model(test_input)
                    # Simulate loss calculation
                    loss = output.view(-1, output.size(-1)).mean()
                    loss.backward()
                
                # Check memory after backward pass
                memory_used_full = torch.cuda.memory_allocated() / 1024**3
                memory_percent_full = (memory_used_full / memory_total) * 100
                
                print(f"   Batch size {batch_size} (with backward): {memory_percent_full:.1f}% GPU memory")
                
                if memory_percent_full > 70:  # Even more conservative with gradients
                    optimal_batch_size = max(batch_size // 2, 1)
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    optimal_batch_size = max(batch_size // 2, 1)
                    break
                else:
                    # Continue without backward test
                    pass
                
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal_batch_size = max(batch_size // 2, 1)
                break
            else:
                raise e
        finally:
            # Clean up
            torch.cuda.empty_cache()
    else:
        optimal_batch_size = batch_size // 2
    
    print(f"✅ Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def gradient_accumulation_training(model, train_loader, optimizer, device, 
                                 effective_batch_size=32, actual_batch_size=8):
    """
    Implement gradient accumulation for larger effective batch sizes
    """
    accumulation_steps = effective_batch_size // actual_batch_size
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        total_loss += loss.item()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            print(f"Effective batch {(batch_idx + 1) // accumulation_steps}: "
                  f"Loss {total_loss:.4f}")
            total_loss = 0


def memory_efficient_dataloader(dataset, batch_size, num_workers=4):
    """
    Create memory-efficient dataloader with optimal settings
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True,          # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2,        # Prefetch batches
        drop_last=True,
        shuffle=True
    )


# Recommended configurations for different GPU memory sizes
GPU_MEMORY_CONFIGS = {
    "4GB": {
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "max_length": 256,
        "effective_batch_size": 32
    },
    "8GB": {
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "effective_batch_size": 32
    },
    "16GB": {
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "max_length": 1024,
        "effective_batch_size": 32
    },
    "24GB+": {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "max_length": 1024,
        "effective_batch_size": 32
    }
}

def get_recommended_config(gpu_memory_gb):
    """Get recommended configuration based on GPU memory"""
    if gpu_memory_gb <= 4:
        return GPU_MEMORY_CONFIGS["4GB"]
    elif gpu_memory_gb <= 8:
        return GPU_MEMORY_CONFIGS["8GB"]
    elif gpu_memory_gb <= 16:
        return GPU_MEMORY_CONFIGS["16GB"]
    else:
        return GPU_MEMORY_CONFIGS["24GB+"]
