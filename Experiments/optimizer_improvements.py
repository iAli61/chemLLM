# Learning Rate and Optimizer Optimization

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup - much better than constant LR
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
                   for base_lr in self.base_lrs]


def setup_optimized_optimizer(model, learning_rate=5e-4, weight_decay=0.1):
    """
    Setup optimizer with layer-wise learning rate decay
    """
    # Separate parameters for different treatment
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Use AdamW with optimized settings
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),  # Slightly adjusted betas
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    return optimizer


def learning_rate_finder(model, train_loader, device, start_lr=1e-7, end_lr=10, num_iter=100):
    """
    Find optimal learning rate using the learning rate range test
    """
    model.train()
    
    # Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(end_lr/start_lr)**(1/num_iter)
    )
    
    losses = []
    lrs = []
    
    # Save initial state
    initial_state = model.state_dict()
    
    print("🔍 Finding optimal learning rate...")
    
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        if batch_idx >= num_iter:
            break
        
        # Forward pass
        optimizer.zero_grad()
        
        # Simple loss calculation (you may need to adjust this based on your model)
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        
        # Store values
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # Stop if loss explodes
        if batch_idx > 10 and losses[-1] > 4 * min(losses):
            break
    
    # Restore initial state
    model.load_state_dict(initial_state)
    
    # Find optimal LR (steepest descent)
    gradients = []
    for i in range(1, len(losses)):
        gradient = (losses[i] - losses[i-1]) / (lrs[i] - lrs[i-1])
        gradients.append(gradient)
    
    min_gradient_idx = gradients.index(min(gradients))
    optimal_lr = lrs[min_gradient_idx + 1]
    
    print(f"✅ Suggested learning rate: {optimal_lr:.2e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(lrs, losses)
    plt.axvline(x=optimal_lr, color='red', linestyle='--', label=f'Suggested LR: {optimal_lr:.2e}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.legend()
    plt.title('Learning Rate Finder')
    
    plt.subplot(1, 2, 2)
    plt.plot(lrs[1:], gradients)
    plt.axvline(x=optimal_lr, color='red', linestyle='--')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss Gradient')
    plt.xscale('log')
    plt.title('Loss Gradient')
    
    plt.tight_layout()
    plt.savefig('lr_finder_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return optimal_lr, losses, lrs


def setup_lr_scheduler(optimizer, num_epochs, steps_per_epoch, warmup_epochs=1):
    """
    Setup learning rate scheduler with warmup
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1
    )
    
    return scheduler


# Advanced optimization techniques
class LayerWiseDecayOptimizer:
    """
    Implements layer-wise learning rate decay (like in BERT)
    """
    def __init__(self, model, base_lr=1e-4, decay_rate=0.8):
        self.optimizers = []
        self.schedulers = []
        
        # Get layer names and assign different learning rates
        layer_lrs = {}
        num_layers = 12  # GPT-2 124M has 12 layers
        
        for name, param in model.named_parameters():
            if 'transformer.h.' in name:
                # Extract layer number
                layer_num = int(name.split('transformer.h.')[1].split('.')[0])
                lr = base_lr * (decay_rate ** (num_layers - layer_num - 1))
            else:
                lr = base_lr
            
            layer_lrs[name] = lr
        
        # Group parameters by learning rate
        lr_groups = {}
        for name, lr in layer_lrs.items():
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(name)
        
        # Create separate optimizers for each learning rate
        for lr, param_names in lr_groups.items():
            params = [p for n, p in model.named_parameters() if n in param_names]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
            self.optimizers.append(optimizer)
    
    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()
    
    def step(self):
        for opt in self.optimizers:
            opt.step()


# Performance monitoring for optimization
class OptimizationTracker:
    def __init__(self):
        self.metrics = {
            'learning_rates': [],
            'grad_norms': [],
            'param_norms': [],
            'loss_values': [],
            'step_times': []
        }
    
    def log_step(self, optimizer, model, loss, step_time):
        # Log learning rates
        lrs = [group['lr'] for group in optimizer.param_groups]
        self.metrics['learning_rates'].append(lrs[0])
        
        # Log gradient norms
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        self.metrics['grad_norms'].append(grad_norm ** 0.5)
        
        # Log parameter norms
        param_norm = 0
        for param in model.parameters():
            param_norm += param.data.norm(2).item() ** 2
        self.metrics['param_norms'].append(param_norm ** 0.5)
        
        # Log loss and timing
        self.metrics['loss_values'].append(loss)
        self.metrics['step_times'].append(step_time)
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].plot(self.metrics['learning_rates'])
        axes[0, 0].set_title('Learning Rate')
        axes[0, 0].set_ylabel('LR')
        
        axes[0, 1].plot(self.metrics['grad_norms'])
        axes[0, 1].set_title('Gradient Norm')
        axes[0, 1].set_ylabel('Norm')
        
        axes[0, 2].plot(self.metrics['param_norms'])
        axes[0, 2].set_title('Parameter Norm')
        axes[0, 2].set_ylabel('Norm')
        
        axes[1, 0].plot(self.metrics['loss_values'])
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_ylabel('Loss')
        
        axes[1, 1].plot(self.metrics['step_times'])
        axes[1, 1].set_title('Step Time')
        axes[1, 1].set_ylabel('Seconds')
        
        # Throughput (tokens per second estimate)
        if self.metrics['step_times']:
            avg_step_time = sum(self.metrics['step_times']) / len(self.metrics['step_times'])
            tokens_per_step = 256 * 2  # batch_size * seq_length
            throughput = tokens_per_step / avg_step_time
            axes[1, 2].text(0.5, 0.5, f'Avg Throughput:\n{throughput:.0f} tokens/sec', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 2].set_title('Throughput')
        
        plt.tight_layout()
        plt.savefig('optimization_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
