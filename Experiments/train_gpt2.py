import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt2model import GPT2
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import time
# env file
from dotenv import load_dotenv
import os
load_dotenv()

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)

DATASET_ID = "iAli61/chempile-education-dedup-dedup"
# download dataset
dataset = load_dataset(DATASET_ID, split="train")

COLUMN = "text"  # Column to be used for deduplication

def text_generation_greedy_V1(model, prompt, tokenizer, context_length=1024, max_new_tokens=50):
    """
    Generate text using greedy decoding.
    
    Args:
        model: The GPT model.
        prompt: The input text prompt.
        tokenizer: The tokenizer for the model.
        context_length: The maximum context length for the model.
        max_new_tokens: The maximum number of new tokens to generate.
        
    Returns:
        Generated text as a string.
    """
    # check if the mode is not gpu, move it to gpu
    if not next(model.parameters()).is_cuda:
        model = model.cuda()
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})).unsqueeze(0).cuda()

    generated_ids = input_ids[:, -(context_length - 1):] # Keep the last context_length - 1 tokens
    generated_ids = generated_ids.cuda()  # Move to GPU if available

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :]  # Get logits for the last token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Greedy decoding
            
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
            
            if generated_ids.size(1) >= context_length:
                break
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return generated_text

class CHEMPILE_DS_V1(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256, stride=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # add "<|endoftext|>" to the end of each text
        self.texts = " ".join([text + "<|endoftext|>" for text in dataset[COLUMN]][:10])
        self.token_ids = tokenizer.encode(self.texts, allowed_special={"<|endoftext|>"})
        self.input_ids = []
        self.target_ids = []

        # using a sliding window approach to create input-target pairs
        for i in range(0, len(self.token_ids) - self.max_length, stride):
            input_ids = self.token_ids[i:i + self.max_length]
            target_ids = self.token_ids[i + 1:i + self.max_length + 1]
            self.input_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(dataset, 
                      tokenizer, 
                      batch_size=8, 
                      max_length=256, 
                      stride=256,
                      drop_last=True,
                      shuffle=True
                      ):

    chempile_ds = CHEMPILE_DS_V1(dataset, tokenizer, max_length=max_length, stride=stride)

    return DataLoader(chempile_ds, 
                      batch_size=batch_size, 
                      drop_last=drop_last,
                      shuffle=shuffle,
                      pin_memory=True,
                      )


tokenizer = tiktoken.get_encoding("gpt2")

# Train/validation ratio
train_val_split = dataset.train_test_split(test_size=0.1, seed=123)
val_dataset = train_val_split['test']
train_dataset = train_val_split['train']
print(f"Dataset length: {len(dataset)}")
print(f"Training data size: {len(train_dataset)}")
print(f"Validation data size: {len(val_dataset)}")


torch.manual_seed(123)

train_loader = create_dataloader(
    train_dataset,
    tokenizer,
    batch_size=2,
    max_length=256,
    stride=256,
    drop_last=True,
    shuffle=True
)

print(f"Number of training batches: {len(train_loader)}")

val_loader = create_dataloader(
    val_dataset,
    tokenizer,
    batch_size=2,
    max_length=256,
    stride=256,
    drop_last=True,
    shuffle=False
)
print(f"Number of validation batches: {len(val_loader)}")

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # Performance tracking variables
    epoch_start_time = time.time()
    total_train_time = 0
    
    print(f"Starting training with {len(train_loader)} batches per epoch...")
    print(f"{'='*60}")

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_start = time.time()
        epoch_tokens = 0
        batch_times = []
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            batch_start = time.time()
            
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            
            batch_tokens = input_batch.numel()
            tokens_seen += batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Print batch-level metrics every 10 batches
            if (batch_idx + 1) % 10 == 0:
                tokens_per_sec = batch_tokens / batch_time
                avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Tokens/sec: {tokens_per_sec:.0f} | "
                      f"Batch time: {batch_time:.3f}s | "
                      f"Avg batch time: {avg_batch_time:.3f}s")

            # Optional evaluation step
            if global_step % eval_freq == 0:
                eval_start = time.time()
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                eval_time = time.time() - eval_start
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"  Evaluation (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} | "
                      f"Eval time: {eval_time:.3f}s")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        tokens_per_sec_epoch = epoch_tokens / epoch_time
        avg_batch_time_epoch = sum(batch_times) / len(batch_times)
        
        print(f"{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Epoch time: {epoch_time:.2f}s")
        print(f"  Tokens processed: {epoch_tokens:,}")
        print(f"  Tokens/sec (epoch): {tokens_per_sec_epoch:.0f}")
        print(f"  Avg batch time: {avg_batch_time_epoch:.3f}s")
        print(f"  Total training time: {total_train_time:.2f}s")
        print(f"  Total tokens seen: {tokens_seen:,}")
        
        # Print a sample text after each epoch
        generation_start = time.time()
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        generation_time = time.time() - generation_start
        print(f"  Text generation time: {generation_time:.3f}s")
        print(f"{'='*60}")

    # Final summary
    avg_tokens_per_sec = tokens_seen / total_train_time
    print(f"\nTraining Complete!")
    print(f"Total training time: {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
    print(f"Total tokens processed: {tokens_seen:,}")
    print(f"Average tokens/sec: {avg_tokens_per_sec:.0f}")
    print(f"Total steps: {global_step + 1}")
    

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.positional_embedding.weight.shape[0]
    with torch.no_grad():
        generated_text = text_generation_greedy_V1(
            model=model, 
            prompt=start_context,
            tokenizer=tokenizer,
            context_length=context_size,
            max_new_tokens=50   
        )
    print(f"Generated text: {generated_text}")

    torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1, fused=None)

num_epochs = 1
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

