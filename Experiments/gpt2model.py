import torch
import torch.nn as nn

class MHA(nn.Module):
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
        
        use_dropout = 0. if self.training else self.config['drop_rate']
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


class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.mha = MHA(config)
        self.ln1 = nn.LayerNorm(config['emb_dim'])
        self.ffn = nn.Sequential(
            nn.Linear(config['emb_dim'], config['emb_dim'] * 4),
            nn.GELU(),
            nn.Linear(config['emb_dim'] * 4, config['emb_dim'])
        )
        self.ln2 = nn.LayerNorm(config['emb_dim'])
        self.drop = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        # Multi-head attention
        x = x + self.drop(self.mha(self.ln1(x)))
        # Feed-forward network
        x = x + self.drop(self.ffn(self.ln2(x)))

        return x
    
class GPT2(torch.nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.positional_embedding = nn.Embedding(config['context_length'], config['emb_dim'])
        self.dropout_emb = nn.Dropout(config['drop_rate'])
        # Using nn.Sequential for transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )
        self.ln_f = nn.LayerNorm(config['emb_dim'])
        self.head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, x):
        B, T = x.size()
        # x: (B, T) where B is batch size and T is sequence length
        # (B, T) -> (B, T, emb_dim)
        x = self.embedding(x) + self.positional_embedding(torch.arange(T, device=x.device))
        x = self.dropout_emb(x)

        # Transformer blocks
        x = self.transformer_blocks(x)

        # Final layer normalization and linear projection
        x = self.ln_f(x)
        logits = self.head(x)

        return logits