# Performance Improvements for GPT-2 Training

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class OptimizedCHEMPILE_DS(Dataset):
    """Optimized dataset with better memory management and caching"""
    
    def __init__(self, dataset, tokenizer, max_length=256, stride=256, num_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Use more data but with better memory management
        if num_samples is None:
            num_samples = min(1000, len(dataset))  # Use more samples
        
        print(f"Processing {num_samples} samples...")
        
        # Process texts in chunks to avoid memory issues
        self.input_ids = []
        self.target_ids = []
        
        chunk_size = 100
        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            chunk_texts = dataset[i:end_idx]['text']
            
            # Process chunk
            chunk_text = " ".join([text + "<|endoftext|>" for text in chunk_texts])
            token_ids = tokenizer.encode(chunk_text, allowed_special={"<|endoftext|>"})
            
            # Create sliding windows for this chunk
            for j in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[j:j + max_length]
                target_chunk = token_ids[j + 1:j + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
        
        print(f"Created {len(self.input_ids)} training samples")

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def optimized_calc_loss_batch(input_batch, target_batch, model, device):
    """Optimized loss calculation with memory efficiency"""
    # Move data to device more efficiently
    if not input_batch.is_cuda:
        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
    
    # Use automatic mixed precision for faster computation
    with torch.amp.autocast(device_type='cuda'):
        logits = model(input_batch)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target_batch.view(-1),
            ignore_index=-100  # Ignore padding tokens if any
        )
    return loss


def train_model_optimized(model, train_loader, val_loader, optimizer, device, num_epochs,
                         eval_freq, eval_iter, start_context, tokenizer):
    """Optimized training loop with multiple performance improvements"""
    
    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # Performance tracking
    total_train_time = 0
    best_val_loss = float('inf')
    
    # Optimization: Compile model for faster execution (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled for faster execution")
    except:
        print("Model compilation not available, continuing without it")
    
    print(f"Starting optimized training with {len(train_loader)} batches per epoch...")
    print(f"Using device: {device}")
    print(f"Mixed precision: Enabled")
    print(f"{'='*70}")

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        epoch_tokens = 0
        epoch_loss = 0
        
        # Use progress tracking
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            batch_start = time.time()
            
            # Optimization: Zero gradients more efficiently
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                loss = optimized_calc_loss_batch(input_batch, target_batch, model, device)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            batch_tokens = input_batch.numel()
            tokens_seen += batch_tokens
            epoch_tokens += batch_tokens
            epoch_loss += loss.item()
            global_step += 1
            
            batch_time = time.time() - batch_start
            
            # More frequent progress updates for better monitoring
            if (batch_idx + 1) % 5 == 0:  # Every 5 batches instead of 10
                tokens_per_sec = batch_tokens / batch_time
                avg_loss = epoch_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / len(train_loader) * 100
                
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} ({progress:.1f}%) | "
                      f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                      f"Tokens/sec: {tokens_per_sec:.0f} | "
                      f"Time: {batch_time:.3f}s")

            # Evaluation with early stopping check
            if global_step % eval_freq == 0:
                eval_start = time.time()
                train_loss, val_loss = evaluate_model_optimized(
                    model, train_loader, val_loader, device, eval_iter)
                eval_time = time.time() - eval_start
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                # Early stopping check
                improvement = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improvement = " (NEW BEST!)"
                    # Save best model
                    torch.save(model.state_dict(), 'best_model.pth')
                
                print(f"  📊 Evaluation (Step {global_step:06d}): "
                      f"Train: {train_loss:.4f}, Val: {val_loss:.4f}{improvement} | "
                      f"Eval time: {eval_time:.2f}s")

        # Epoch summary with detailed metrics
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        tokens_per_sec_epoch = epoch_tokens / epoch_time
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        print(f"{'='*70}")
        print(f"🎯 Epoch {epoch+1}/{num_epochs} Complete:")
        print(f"   ⏱️  Time: {epoch_time:.2f}s")
        print(f"   🔢 Tokens: {epoch_tokens:,} ({tokens_per_sec_epoch:.0f} tokens/sec)")
        print(f"   📉 Avg Loss: {avg_epoch_loss:.4f}")
        print(f"   💾 Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else "")
        print(f"   📊 Total Time: {total_train_time:.2f}s | Total Tokens: {tokens_seen:,}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate sample text
        generation_start = time.time()
        generate_and_print_sample_optimized(model, tokenizer, device, start_context)
        generation_time = time.time() - generation_start
        print(f"   🤖 Generation time: {generation_time:.3f}s")
        print(f"{'='*70}")

    # Final performance summary
    avg_tokens_per_sec = tokens_seen / total_train_time
    efficiency_score = (tokens_seen / 1000) / (total_train_time / 60)  # K tokens per minute
    
    print(f"\n🏁 Training Complete! Performance Summary:")
    print(f"   ⏱️  Total time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
    print(f"   🔢 Total tokens: {tokens_seen:,}")
    print(f"   🚀 Average speed: {avg_tokens_per_sec:.0f} tokens/sec")
    print(f"   📈 Efficiency: {efficiency_score:.1f}K tokens/min")
    print(f"   📊 Steps completed: {global_step + 1}")
    print(f"   🎯 Best validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses, track_tokens_seen


def evaluate_model_optimized(model, train_loader, val_loader, device, eval_iter):
    """Optimized evaluation with mixed precision"""
    model.eval()
    with torch.no_grad():
        # Use mixed precision for evaluation too
        with torch.cuda.amp.autocast():
            train_loss = calc_loss_loader_optimized(train_loader, model, device, eval_iter)
            val_loss = calc_loss_loader_optimized(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def calc_loss_loader_optimized(data_loader, model, device, num_batches=None):
    """Optimized loss calculation for data loader"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        with torch.cuda.amp.autocast():
            loss = optimized_calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    
    return total_loss / num_batches


def generate_and_print_sample_optimized(model, tokenizer, device, start_context):
    """Optimized text generation"""
    model.eval()
    context_size = model.positional_embedding.weight.shape[0]
    
    with torch.no_grad():
        # Use mixed precision for generation
        with torch.cuda.amp.autocast():
            generated_text = text_generation_greedy_optimized(
                model=model, 
                prompt=start_context,
                tokenizer=tokenizer,
                context_length=context_size,
                max_new_tokens=50   
            )
    print(f"   💬 Generated: {generated_text[:100]}...")


def text_generation_greedy_optimized(model, prompt, tokenizer, context_length=1024, max_new_tokens=50):
    """Optimized text generation with better memory management"""
    model.eval()
    
    # Efficient tokenization
    input_ids = torch.tensor(
        tokenizer.encode(prompt, allowed_special={"<|endoftext|>"}),
        dtype=torch.long,
        device=next(model.parameters()).device
    ).unsqueeze(0)

    # Keep only the relevant context
    if input_ids.size(1) > context_length - max_new_tokens:
        input_ids = input_ids[:, -(context_length - max_new_tokens):]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            
            if input_ids.size(1) >= context_length:
                break
    
    return tokenizer.decode(input_ids[0].tolist())


# Additional optimization utilities
class PerformanceProfiler:
    """Simple profiler to track training bottlenecks"""
    
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def start(self, name):
        self.times[name] = time.time()
    
    def end(self, name):
        if name in self.times:
            duration = time.time() - self.times[name]
            if name not in self.counts:
                self.counts[name] = []
            self.counts[name].append(duration)
    
    def report(self):
        print("\n📊 Performance Profiling Report:")
        for name, times in self.counts.items():
            avg_time = np.mean(times)
            total_time = np.sum(times)
            print(f"   {name}: {avg_time:.4f}s avg, {total_time:.2f}s total ({len(times)} calls)")


# Memory optimization utilities
def optimize_memory_usage():
    """Configure PyTorch for optimal memory usage"""
    if torch.cuda.is_available():
        # Enable memory-efficient attention if available
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except AttributeError:
            try:
                torch.backends.cuda.enable_memory_efficient_sdp(True)
            except AttributeError:
                print("⚠️  Memory efficient attention not available")
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        print("✅ CUDA memory optimizations enabled")


def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb
