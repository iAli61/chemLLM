# Complete Training Script
# This integrates all performance improvements into a single, production-ready script
# Note: Imports optimized functions from performance modules but exposes clean API

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
from datasets import load_dataset
import tiktoken
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Import our optimization modules
from performance_improvements import *
from batch_optimization import *
from optimizer_improvements import *
from model_optimizations import *

load_dotenv()

# Configuration for 6GB GPU - smaller model
CONFIG_6GB = {
    "vocab_size": 50257,
    "context_length": 256,  # Reduced for memory
    "emb_dim": 512,         # Reduced from 768
    "n_heads": 8,           # Reduced from 12
    "n_layers": 8,          # Reduced from 12
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Original larger config (for reference)
CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class TrainingPipeline:
    """
    Complete training pipeline with all performance improvements
    """
    
    def __init__(self, config, dataset_id="iAli61/chempile-education-dedup-dedup"):
        self.config = config
        self.dataset_id = dataset_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        # Initialize mixed precision scaler
        if torch.cuda.is_available():
            try:
                # Try new API first (PyTorch 2.0+)
                self.scaler = torch.amp.GradScaler('cuda')
            except (AttributeError, TypeError):
                # Fallback to older API
                self.scaler = torch.amp.GradScaler()
        else:
            # CPU fallback (no mixed precision)
            self.scaler = None
        
        # Performance tracking
        self.profiler = PerformanceProfiler()
        self.tracker = OptimizationTracker()
        
        print(f"🚀 Initialized Training Pipeline")
        print(f"   Device: {self.device}")
        print(f"   Config: {config}")
    
    def setup_model_and_optimizer(self, learning_rate=None):
        """Setup model and optimizer"""
        print("🔧 Setting up model and optimizer...")
        
        # Enable PyTorch optimizations
        enable_torch_optimizations()
        optimize_memory_usage()
        
        # Initialize model
        self.model = OptimizedGPT2(self.config).to(self.device)
        
        # Model compilation for extra speed (PyTorch 2.0+)
        # Disable compilation temporarily to avoid CUDA graph issues
        compile_enabled = False  # Set to True if you want to test compilation
        
        if compile_enabled:
            try:
                # Use a more conservative compilation mode
                self.model = torch.compile(self.model, mode='default', dynamic=True)
                print("✅ Model compiled for performance")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {str(e)[:100]}...")
                print("   Continuing without compilation")
        else:
            print("⚠️  Model compilation disabled (avoiding CUDA graph issues)")
        
        # Print model summary
        model_summary(self.model)
        
        # Find optimal learning rate if not provided
        if learning_rate is None:
            print("🔍 Finding optimal learning rate...")
            # We'll set a default for now, but you can uncomment the LR finder
            learning_rate = 3e-4
            # learning_rate = self.find_optimal_lr()
        
        # Setup optimizer
        self.optimizer = self.model.configure_optimizers(
            learning_rate=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95)
        )
        
        print(f"✅ Model and optimizer ready (LR: {learning_rate:.2e})")
    
    def setup_data(self, num_samples=1000, batch_size=None):
        """Setup data loading"""
        print("📊 Setting up data loading...")
        
        # Load dataset
        dataset = load_dataset(self.dataset_id, split="train")
        
        # Split data
        train_val_split = dataset.train_test_split(test_size=0.1, seed=123)
        train_dataset = train_val_split['train']
        val_dataset = train_val_split['test']
        
        print(f"   Dataset size: {len(dataset):,}")
        print(f"   Train size: {len(train_dataset):,}")
        print(f"   Val size: {len(val_dataset):,}")
        
        # Determine optimal batch size if not provided
        if batch_size is None:
            if torch.cuda.is_available():
                # For 6GB GPU, use very conservative batch size
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < 8:  # Less than 8GB
                    batch_size = 4  # Very conservative for 6GB GPU
                    print(f"🔧 Using conservative batch size {batch_size} for {gpu_memory:.1f}GB GPU")
                    
                    # Clear memory before starting
                    torch.cuda.empty_cache()
                else:
                    # Create sample data for batch size detection
                    sample_input = torch.randint(0, 50257, (1, self.config["context_length"]), device=self.device)
                    batch_size = find_optimal_batch_size(
                        self.model, self.tokenizer, sample_input, 
                        self.device, self.config["context_length"]
                    )
            else:
                batch_size = 4
        
        # Create datasets
        train_ds = OptimizedCHEMPILE_DS(
            train_dataset, self.tokenizer, 
            max_length=self.config["context_length"],
            num_samples=num_samples
        )
        
        val_ds = OptimizedCHEMPILE_DS(
            val_dataset, self.tokenizer,
            max_length=self.config["context_length"],
            num_samples=num_samples // 10
        )
        
        # Create data loaders
        self.train_loader = memory_efficient_dataloader(train_ds, batch_size)
        self.val_loader = memory_efficient_dataloader(val_ds, batch_size)
        
        print(f"✅ Data loading ready")
        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        
        return batch_size
    
    def setup_scheduler(self, num_epochs):
        """Setup learning rate scheduler"""
        total_steps = num_epochs * len(self.train_loader)
        warmup_steps = len(self.train_loader)  # 1 epoch warmup
        
        self.scheduler = setup_lr_scheduler(
            self.optimizer, num_epochs, len(self.train_loader), warmup_epochs=1
        )
        
        print(f"📈 Scheduler setup: {warmup_steps} warmup steps, {total_steps} total steps")
    
    def train(self, num_epochs=10, eval_freq=50, save_freq=1000):
        """
        Training loop with all performance improvements
        """
        print(f"🎯 Starting training for {num_epochs} epochs...")
        print(f"{'='*80}")
        
        # Setup scheduler
        self.setup_scheduler(num_epochs)
        
        # Training tracking
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, 0
        total_train_time = 0
        best_val_loss = float('inf')
        
        # Start training
        for epoch in range(num_epochs):
            self.model.train()
            epoch_start = time.time()
            epoch_tokens = 0
            epoch_loss = 0
            
            print(f"🚀 Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (input_batch, target_batch) in enumerate(self.train_loader):
                step_start = time.time()
                
                # Memory monitoring for 6GB GPU
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    if memory_used > 5.0:  # If using more than 5GB, warn
                        print(f"⚠️  High GPU memory usage: {memory_used:.2f}GB")
                        torch.cuda.empty_cache()  # Try to free some memory
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad(set_to_none=True)
                
                # Use mixed precision if available
                if self.scaler is not None and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        loss = optimized_calc_loss_batch(
                            input_batch, target_batch, self.model, self.device
                        )
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping for stability
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # CPU or no mixed precision
                    loss = optimized_calc_loss_batch(
                        input_batch, target_batch, self.model, self.device
                    )
                    
                    # Standard backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                self.scheduler.step()
                
                # Track metrics
                batch_tokens = input_batch.numel()
                tokens_seen += batch_tokens
                epoch_tokens += batch_tokens
                epoch_loss += loss.item()
                global_step += 1
                
                step_time = time.time() - step_start
                
                # Log metrics
                self.tracker.log_step(
                    self.optimizer, self.model, loss.item(), step_time
                )
                
                # Progress reporting
                if global_step % 20 == 0:
                    tokens_per_sec = batch_tokens / step_time
                    avg_loss = epoch_loss / (batch_idx + 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    print(f"   Step {global_step:05d} | "
                          f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                          f"LR: {lr:.2e} | "
                          f"Speed: {tokens_per_sec:.0f} tok/s | "
                          f"Time: {step_time:.3f}s")
                
                # Evaluation
                if global_step % eval_freq == 0:
                    eval_start = time.time()
                    train_loss, val_loss = self.evaluate()
                    eval_time = time.time() - eval_start
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    
                    improvement = ""
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        improvement = " 🎯 NEW BEST!"
                        self.save_checkpoint('best_model.pth')
                    
                    print(f"   📊 Eval (Step {global_step:05d}): "
                          f"Train: {train_loss:.4f}, Val: {val_loss:.4f}{improvement} | "
                          f"Time: {eval_time:.2f}s")
                
                # Save checkpoint
                if global_step % save_freq == 0:
                    self.save_checkpoint(f'checkpoint_step_{global_step}.pth')
                    print(f"   💾 Checkpoint saved at step {global_step}")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            tokens_per_sec_epoch = epoch_tokens / epoch_time
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            
            print(f"{'='*80}")
            print(f"📈 Epoch {epoch+1} Summary:")
            print(f"   ⏱️  Time: {epoch_time:.2f}s")
            print(f"   🔢 Tokens: {epoch_tokens:,} ({tokens_per_sec_epoch:.0f} tok/s)")
            print(f"   📉 Loss: {avg_epoch_loss:.4f}")
            print(f"   💾 GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else "")
            
            # Generate sample
            if epoch % 2 == 0:  # Every 2 epochs
                self.generate_sample("The chemical compound")
            
            print(f"{'='*80}")
        
        # Final summary
        avg_speed = tokens_seen / total_train_time
        print(f"\n🏁 Training Complete!")
        print(f"   ⏱️  Total time: {total_train_time/60:.2f} minutes")
        print(f"   🔢 Total tokens: {tokens_seen:,}")
        print(f"   🚀 Average speed: {avg_speed:.0f} tokens/sec")
        print(f"   🎯 Best val loss: {best_val_loss:.4f}")
        
        # Plot training metrics
        self.plot_training_results(train_losses, val_losses, track_tokens_seen)
        self.tracker.plot_metrics()
        
        return train_losses, val_losses, track_tokens_seen
    
    def evaluate(self, num_batches=5):
        """Evaluation"""
        self.model.eval()
        
        with torch.no_grad():
            train_loss = calc_loss_loader_optimized(
                self.train_loader, self.model, self.device, num_batches
            )
            val_loss = calc_loss_loader_optimized(
                self.val_loader, self.model, self.device, num_batches
            )
        
        self.model.train()
        return train_loss, val_loss
    
    def generate_sample(self, prompt="Every effort moves you", max_tokens=50):
        """Generate sample text"""
        self.model.eval()
        
        with torch.no_grad():
            generated = text_generation_greedy_optimized(
                self.model, prompt, self.tokenizer,
                self.config["context_length"], max_tokens
            )
        
        print(f"   🤖 Sample: {generated[:150]}...")
        self.model.train()
    
    def save_checkpoint(self, filename):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Load training checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"✅ Checkpoint loaded from {filename}")
    
    def plot_training_results(self, train_losses, val_losses, tokens_seen):
        """Plot comprehensive training results"""
        if not train_losses:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Evaluation Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss vs tokens
        if tokens_seen:
            axes[0, 1].plot(tokens_seen, train_losses, label='Train Loss', color='blue')
            axes[0, 1].plot(tokens_seen, val_losses, label='Val Loss', color='red')
            axes[0, 1].set_title('Loss vs Tokens Seen')
            axes[0, 1].set_xlabel('Tokens Processed')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate schedule
        if hasattr(self.tracker, 'metrics'):
            lrs = self.tracker.metrics.get('learning_rates', [])
            if lrs:
                axes[1, 0].plot(lrs)
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].grid(True)
        
        # Performance metrics
        step_times = self.tracker.metrics.get('step_times', [])
        if step_times:
            # Calculate throughput
            throughput = [512 * 8 / t for t in step_times]  # tokens per second estimate
            axes[1, 1].plot(throughput)
            axes[1, 1].set_title('Training Throughput')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Tokens/Second')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main training function"""
    print("🚀 Starting GPT-2 Training (6GB GPU)")
    print("="*80)
    
    # Use smaller config for 6GB GPU
    config = CONFIG_6GB
    print(f"📏 Using 6GB GPU config:")
    print(f"   Model size: ~{config['emb_dim'] * config['n_layers'] / 1000:.0f}K parameters per layer")
    print(f"   Context length: {config['context_length']}")
    print(f"   Embedding dim: {config['emb_dim']}")
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(config)
    
    # Setup model and optimizer
    pipeline.setup_model_and_optimizer(learning_rate=3e-4)
    
    # Setup data (use fewer samples for 6GB GPU)
    batch_size = pipeline.setup_data(num_samples=2000, batch_size=None)  # Reduced samples
    
    # Start training
    train_losses, val_losses, tokens_seen = pipeline.train(
        num_epochs=5,  # Reduced for demo
        eval_freq=50,
        save_freq=500
    )
    
    print("\n🎉 Training completed successfully!")
    print("📊 Check the generated plots for detailed performance analysis")
    
    return pipeline, train_losses, val_losses, tokens_seen


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(123)
    
    # Run training
    pipeline, train_losses, val_losses, tokens_seen = main()
