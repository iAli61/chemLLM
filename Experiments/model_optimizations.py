# Model Architecture and Implementation Optimizations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class OptimizedGPT2Attention(nn.Module):
    """
    Optimized attention implementation with Flash Attention and other improvements
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["emb_dim"]
        self.n_heads = config["n_heads"]
        self.head_dim = self.embed_dim // self.n_heads
        
        # Use single linear layer for better efficiency
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.get("qkv_bias", False))
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout = nn.Dropout(config["drop_rate"])
        self.register_buffer("mask", torch.triu(torch.ones(config["context_length"], 
                                                         config["context_length"]), diagonal=1))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Single QKV projection for efficiency
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch, heads, seq, 3*head_dim)
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Use Flash Attention when available (PyTorch 2.0+)
     
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0,
            is_causal=True  # Much more efficient than manual masking
        )

        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)
    
    def _manual_attention(self, q, k, v, seq_len):
        """Fallback manual attention implementation"""
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.mask[:seq_len, :seq_len]
        attn_weights.masked_fill_(mask.bool(), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)


class OptimizedGPT2MLP(nn.Module):
    """
    Optimized MLP with better activation and initialization
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["emb_dim"], 4 * config["emb_dim"])
        self.fc2 = nn.Linear(4 * config["emb_dim"], config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization"""
        for module in [self.fc1, self.fc2]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Use GELU activation (better than ReLU for transformers)
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')  # Faster approximation
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class OptimizedGPT2Block(nn.Module):
    """
    Optimized transformer block with pre-normalization and better residual connections
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["emb_dim"])
        self.attn = OptimizedGPT2Attention(config)
        self.ln2 = nn.LayerNorm(config["emb_dim"])
        self.mlp = OptimizedGPT2MLP(config)
        
        # Learnable scaling factors for residual connections
        self.ls1 = nn.Parameter(torch.ones(config["emb_dim"]) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(config["emb_dim"]) * 0.1)
    
    def forward(self, x):
        # Pre-normalization (more stable training)
        x = x + self.ls1 * self.attn(self.ln1(x))
        x = x + self.ls2 * self.mlp(self.ln2(x))
        return x


class OptimizedGPT2(nn.Module):
    """
    Fully optimized GPT-2 model with all performance improvements
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.positional_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.emb_dropout = nn.Dropout(config["drop_rate"])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            OptimizedGPT2Block(config) for _ in range(config["n_layers"])
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        
        # Tie embeddings (reduce parameters)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale initialization for residual connections
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('fc2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config["n_layers"]))
    
    def _init_weights(self, module):
        """Improved weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(pos)
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def get_num_params(self):
        """Get number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def configure_optimizers(self, learning_rate, weight_decay, betas):
        """
        Configure optimizers with parameter-specific settings
        """
        # Separate decay and no_decay parameters
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn in ['ls1', 'ls2']:  # Add learnable scaling parameters to no_decay
                    no_decay.add(fpn)

        # Get actual parameters (this handles weight tying correctly)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Remove any parameter names that don't exist due to weight tying
        decay = decay.intersection(param_dict.keys())
        no_decay = no_decay.intersection(param_dict.keys())
        
        # Validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"
        
        # Create the optimizer
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


# Additional optimization utilities
def setup_mixed_precision_training():
    """Setup for mixed precision training"""
    scaler = torch.cuda.amp.GradScaler()
    return scaler


def enable_torch_optimizations():
    """Enable various PyTorch optimizations"""
    # Optimize for inference/training
    torch.backends.cudnn.benchmark = True
    
    # Enable memory-efficient attention (try different API names for compatibility)
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except AttributeError:
        try:
            torch.backends.cuda.enable_memory_efficient_sdp(True)
        except AttributeError:
            print("⚠️  Memory efficient attention not available")
    
    # Enable flash attention (try different API names for compatibility)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        try:
            torch.backends.cuda.enable_flash_attention(True)
        except AttributeError:
            print("⚠️  Flash attention backend not available")
    
    print("✅ PyTorch optimizations enabled")


def model_summary(model, input_size=(2, 256)):
    """Print detailed model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB (fp32)")
    print(f"  Model size: {total_params * 2 / 1024**2:.2f} MB (fp16)")
    
    # Memory usage estimation
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        sample_input = torch.randint(0, 1000, input_size, device=device)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(sample_input)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU memory (forward): {peak_memory:.2f} MB")


# Performance comparison utilities
def benchmark_model_variants(config, input_size=(8, 256), num_runs=10):
    """Benchmark different model implementations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Only benchmark the optimized model since GPT2 class might not be available
    models = {
        "Optimized": OptimizedGPT2(config)
    }
    
    results = {}
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        sample_input = torch.randint(0, config["vocab_size"], input_size, device=device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(sample_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = (input_size[0] * input_size[1]) / avg_time
        
        results[name] = {
            "avg_time": avg_time,
            "throughput": throughput,
            "params": sum(p.numel() for p in model.parameters())
        }
        
        print(f"{name} Model:")
        print(f"  Avg time: {avg_time:.4f}s")
        print(f"  Throughput: {throughput:.0f} tokens/sec")
        print(f"  Parameters: {results[name]['params']:,}")
        print()
    
    # Calculate speedup
    if "Original" in results and "Optimized" in results:
        speedup = results["Original"]["avg_time"] / results["Optimized"]["avg_time"]
        print(f"🚀 Speedup: {speedup:.2f}x faster")
    
    return results
