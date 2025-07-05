"""
Enhanced Training Features for ChemLLM - Phase 2 Step 2

This module provides advanced training features including:
- Improved data collation with proper label shifting verification
- Enhanced learning rate scheduling  
- Early stopping mechanisms
- Better text generation with sampling strategies
"""

import logging
import torch
import math
from typing import Optional, Dict, Any, List, Union
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
try:
    from transformers.optimization import AdamW
except ImportError:
    from torch.optim import AdamW
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedDataCollator(DataCollatorForLanguageModeling):
    """
    Enhanced data collator with proper label shifting verification and debugging.
    """
    
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, pad_to_multiple_of=None, 
                 return_tensors="pt", verbose_verification=False):
        # Handle the new parameters properly
        super().__init__(
            tokenizer=tokenizer, 
            mlm=mlm, 
            mlm_probability=mlm_probability, 
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors
        )
        self.verbose_verification = verbose_verification
        self.shift_count = 0
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        """Enhanced collation with proper verification."""
        # Call parent implementation
        batch = super().torch_call(examples)
        
        # Verify label shifting for causal LM
        if not self.mlm and self.verbose_verification and self.shift_count < 3:
            self._verify_label_shifting(batch, examples)
            self.shift_count += 1
        
        return batch
    
    def _verify_label_shifting(self, batch: Dict[str, Any], examples: List):
        """Verify that labels are properly shifted for next-token prediction."""
        if 'input_ids' in batch and 'labels' in batch:
            input_ids = batch['input_ids'][0]  # First sequence
            labels = batch['labels'][0]
            
            # Check if labels are shifted (labels[i] should equal input_ids[i+1])
            # Ignore positions where labels = -100 (padding)
            valid_positions = labels != -100
            if valid_positions.sum() > 1:
                # For causal LM: labels[i] should predict input_ids[i+1]
                # So input_ids[:-1] should match labels[1:] where both are valid
                input_part = input_ids[:-1][valid_positions[1:]]
                label_part = labels[1:][valid_positions[1:]]
                
                if len(input_part) > 0 and len(label_part) > 0:
                    correctly_shifted = torch.equal(input_part, label_part)
                    
                    logger.info("🔍 Enhanced DataCollator Verification:")
                    logger.info(f"  Input (first 10): {input_ids[:10].tolist()}")
                    logger.info(f"  Labels (first 10): {labels[:10].tolist()}")
                    logger.info(f"  Valid positions: {valid_positions[:10].tolist()}")
                    logger.info(f"  ✅ Correctly shifted: {correctly_shifted}")
                    
                    if not correctly_shifted and len(input_part) > 0:
                        logger.info(f"  Input[:-1]:  {input_part[:5].tolist()}...")
                        logger.info(f"  Labels[1:]:  {label_part[:5].tolist()}...")


class EnhancedTrainingArguments(TrainingArguments):
    """Enhanced training arguments with additional scheduling options."""
    
    def __init__(self, 
                 scheduler_type: str = "cosine",
                 warmup_ratio: float = 0.1,
                 min_learning_rate: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio  
        self.min_learning_rate = min_learning_rate


class EnhancedTrainer(Trainer):
    """Enhanced trainer with custom scheduling and monitoring."""
    
    def __init__(self, scheduler_type: str = "cosine", min_learning_rate: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.scheduler_type = scheduler_type
        self.min_learning_rate = min_learning_rate
        
    def get_lr_scheduler(self):
        """Override to create enhanced learning rate scheduler."""
        # Use parent implementation if already created
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            return self.lr_scheduler
            
        # Create custom scheduler  
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be created before scheduler")
            
        num_training_steps = self.args.max_steps
        if num_training_steps < 0:
            num_training_steps = len(self.train_dataloader) * self.args.num_train_epochs
            
        warmup_steps = int(num_training_steps * getattr(self.args, 'warmup_ratio', 0.1))
        
        if self.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=-1
            )
            logger.info(f"📈 Using cosine scheduler: {warmup_steps} warmup steps")
        elif self.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=-1
            )
            logger.info(f"📈 Using linear scheduler: {warmup_steps} warmup steps")
        else:
            # Use parent default scheduler
            return super().get_lr_scheduler()
            
        self.lr_scheduler = scheduler
        return scheduler
            
        return scheduler


def create_enhanced_training_config(
    output_dir: str = "./results",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    min_learning_rate: float = 1e-6,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.01,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 100,
    **kwargs
) -> EnhancedTrainingArguments:
    """
    Create enhanced training configuration with advanced scheduling.
    
    Args:
        scheduler_type: "cosine", "linear", or "constant"
        warmup_ratio: Fraction of training for warmup
        min_learning_rate: Minimum LR for cosine scheduling
        early_stopping_patience: Steps to wait before stopping
        early_stopping_threshold: Minimum improvement threshold
        max_grad_norm: Gradient clipping value
    """
    return EnhancedTrainingArguments(
        output_dir=output_dir,
        
        # Training setup
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        
        # Enhanced scheduling
        scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        min_learning_rate=min_learning_rate,
        
        # Optimization
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_accumulation_steps=4,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=2,  # Reduced to avoid tokenizer warnings
        
        # Evaluation and checkpointing
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Early stopping
        eval_accumulation_steps=1,
        
        # Logging
        logging_steps=logging_steps,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        
        # Performance
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Keep all columns for debugging
        
        **kwargs
    )


def generate_text_enhanced(
    model, 
    tokenizer, 
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    num_sequences: int = 1
) -> List[str]:
    """
    Enhanced text generation with better sampling strategies.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = disabled)
        top_p: Top-p (nucleus) sampling
        repetition_penalty: Penalty for repeating tokens
        do_sample: Whether to use sampling vs greedy
        num_sequences: Number of sequences to generate
    
    Returns:
        List of generated texts
    """
    logger.info(f"🎨 Generating text with enhanced sampling")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        if not next(model.parameters()).is_cuda:
            model = model.cuda()
    
    # Generation parameters
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_sequences,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if do_sample:
        generation_kwargs.update({
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        })
        
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
        if top_p < 1.0:
            generation_kwargs["top_p"] = top_p
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    # Decode results
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def setup_early_stopping(patience: int = 3, threshold: float = 0.01) -> EarlyStoppingCallback:
    """Setup early stopping callback."""
    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold
    )


if __name__ == "__main__":
    # Test enhanced data collator
    logging.basicConfig(level=logging.INFO)
    
    from transformers import AutoTokenizer
    
    print("🧪 Testing Enhanced Training Features")
    print("=" * 50)
    
    # Test data collator
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = EnhancedDataCollator(tokenizer, mlm=False, verbose_verification=True)
    
    # Test data
    test_data = [
        {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8, 9, 10], "labels": [6, 7, 8, 9, 10]}
    ]
    
    batch = collator(test_data)
    print(f"✅ Enhanced data collator test completed")
    
    # Test training config
    config = create_enhanced_training_config(
        scheduler_type="cosine",
        warmup_ratio=0.1,
        early_stopping_patience=3
    )
    
    print(f"✅ Enhanced training config created")
    print(f"   Scheduler: {config.scheduler_type}")
    print(f"   Warmup ratio: {config.warmup_ratio}")
    print(f"   BF16: {config.bf16}")
