"""
Performance Monitoring for ChemLLM - Phase 2 Step 4

This module provides comprehensive performance monitoring including:
- Weights & Biases integration
- Training metrics dashboard
- Performance benchmarking
- Experiment tracking and comparison
"""

import logging
import time
import json
import psutil
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Optional W&B import with graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("📊 Weights & Biases not available - install with: pip install wandb")


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics tracking."""
    step: int
    epoch: float
    learning_rate: float
    train_loss: float
    eval_loss: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    cpu_percent: Optional[float] = None
    samples_per_second: Optional[float] = None
    tokens_per_second: Optional[float] = None
    grad_norm: Optional[float] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass 
class BenchmarkResults:
    """Performance benchmark results."""
    model_name: str
    batch_size: int
    sequence_length: int
    precision: str
    flash_attention: bool
    gradient_checkpointing: bool
    
    # Performance metrics
    training_speed_samples_per_sec: float
    training_speed_tokens_per_sec: float
    memory_usage_gb: float
    peak_memory_gb: float
    
    # Training quality
    final_train_loss: float
    final_eval_loss: float
    convergence_steps: int
    
    # Hardware info
    gpu_model: str
    gpu_memory_total_gb: float
    cpu_cores: int
    
    timestamp: str


class PerformanceMonitor:
    """Comprehensive performance monitoring with W&B integration."""
    
    def __init__(self, 
                 project_name: str = "chemllm-training",
                 experiment_name: Optional[str] = None,
                 enable_wandb: bool = True,
                 log_frequency: int = 10,
                 batch_size: int = 4,
                 sequence_length: int = 256):
        self.project_name = project_name
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.log_frequency = log_frequency
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = time.time()
        self.step_times = []
        self.tokens_processed = 0
        self.total_training_time = 0
        
        # Initialize W&B if available
        if self.enable_wandb:
            try:
                wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    reinit=True
                )
                logger.info(f"📊 Weights & Biases initialized: {self.project_name}/{self.experiment_name}")
            except Exception as e:
                logger.warning(f"⚠️ W&B initialization failed: {e}")
                self.enable_wandb = False
        
        logger.info(f"🔍 Performance monitoring initialized: {self.experiment_name}")
    
    def log_system_info(self, model, tokenizer, training_args):
        """Log system and model configuration."""
        system_info = {
            # System info
            "cpu_cores": psutil.cpu_count(),
            "cpu_freq_ghz": psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else None,
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            
            # Model info
            "model_name": getattr(model, 'name_or_path', 'unknown'),
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vocab_size": tokenizer.vocab_size,
            
            # Training config
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "precision": str(training_args.bf16 or training_args.fp16 or "fp32"),
            "gradient_checkpointing": getattr(model, 'gradient_checkpointing', False),
        }
        
        # GPU info
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "gpu_count": torch.cuda.device_count(),
            }
            system_info.update(gpu_info)
        
        if self.enable_wandb:
            wandb.config.update(system_info)
        
        logger.info(f"💻 System info logged: {system_info['model_parameters']:,} parameters, {system_info.get('gpu_model', 'CPU only')}")
        return system_info
    
    def log_metrics(self, step: int, epoch: float, metrics: Dict[str, Any], batch_info: Optional[Dict[str, Any]] = None):
        """Log training metrics with system monitoring and tokens per second calculation."""
        step_start_time = time.time()
        
        # Extract core metrics
        train_loss = metrics.get('loss', metrics.get('train_loss'))
        eval_loss = metrics.get('eval_loss')
        learning_rate = metrics.get('learning_rate')
        grad_norm = metrics.get('grad_norm')
        
        # System metrics
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        
        cpu_percent = psutil.cpu_percent()
        
        # Calculate speeds with actual batch information
        samples_per_second = None
        tokens_per_second = None
        
        # Use batch info if provided, otherwise fall back to defaults
        current_batch_size = batch_info.get('batch_size', self.batch_size) if batch_info else self.batch_size
        current_seq_length = batch_info.get('sequence_length', self.sequence_length) if batch_info else self.sequence_length
        
        if len(self.step_times) > 0:
            # Use recent step times for more accurate calculation
            recent_step_times = self.step_times[-5:] if len(self.step_times) >= 5 else self.step_times
            avg_step_time = np.mean(recent_step_times)
            
            if avg_step_time > 0:
                samples_per_second = current_batch_size / avg_step_time
                tokens_per_second = (current_batch_size * current_seq_length) / avg_step_time
                
                # Update tokens processed counter
                self.tokens_processed += current_batch_size * current_seq_length
        
        # Create metrics object
        training_metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            learning_rate=learning_rate,
            train_loss=train_loss,
            eval_loss=eval_loss,
            gpu_memory_gb=gpu_memory,
            cpu_percent=cpu_percent,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            grad_norm=grad_norm
        )
        
        self.metrics_history.append(training_metrics)
        
        # Log to W&B
        if self.enable_wandb and step % self.log_frequency == 0:
            wandb_metrics = {k: v for k, v in asdict(training_metrics).items() 
                           if v is not None and k != 'timestamp'}
            wandb.log(wandb_metrics, step=step)
        
        # Console logging for important metrics with tokens per second
        if step % (self.log_frequency) == 0:  # Log more frequently to see tokens/sec
            tokens_per_sec_str = f", Tokens/sec={tokens_per_second:.0f}" if tokens_per_second else ""
            samples_per_sec_str = f", Samples/sec={samples_per_second:.1f}" if samples_per_second else ""
            logger.info(f"📊 Step {step}: Loss={train_loss:.4f}, LR={learning_rate:.2e}, "
                       f"GPU={gpu_memory:.2f}GB, CPU={cpu_percent:.1f}%{tokens_per_sec_str}{samples_per_sec_str}")
        
        # Track step timing
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        self.step_times.append(step_duration)
        self.total_training_time += step_duration
        
        return training_metrics
    
    def benchmark_model(self, model, tokenizer, test_batch, config: Dict[str, Any]) -> BenchmarkResults:
        """Run comprehensive model benchmarking."""
        logger.info("🏃‍♂️ Starting model benchmarking...")
        
        # Warm up
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = model(**test_batch)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark inference speed
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with torch.no_grad():
            for _ in range(10):
                outputs = model(**test_batch)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        # Calculate metrics
        total_time = end_time - start_time
        batch_size = test_batch['input_ids'].shape[0]
        sequence_length = test_batch['input_ids'].shape[1]
        
        samples_per_sec = (10 * batch_size) / total_time
        tokens_per_sec = (10 * batch_size * sequence_length) / total_time
        memory_usage_gb = (end_memory - start_memory) / (1024**3)
        peak_memory_gb = peak_memory / (1024**3)
        
        # System info
        gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        
        benchmark_results = BenchmarkResults(
            model_name=config.get('model_name', 'unknown'),
            batch_size=batch_size,
            sequence_length=sequence_length,
            precision=config.get('precision', 'unknown'),
            flash_attention=config.get('flash_attention', False),
            gradient_checkpointing=config.get('gradient_checkpointing', False),
            
            training_speed_samples_per_sec=samples_per_sec,
            training_speed_tokens_per_sec=tokens_per_sec,
            memory_usage_gb=memory_usage_gb,
            peak_memory_gb=peak_memory_gb,
            
            final_train_loss=config.get('final_train_loss', 0.0),
            final_eval_loss=config.get('final_eval_loss', 0.0),
            convergence_steps=config.get('convergence_steps', 0),
            
            gpu_model=gpu_model,
            gpu_memory_total_gb=gpu_memory_total_gb,
            cpu_cores=psutil.cpu_count(),
            
            timestamp=datetime.now().isoformat()
        )
        
        if self.enable_wandb:
            wandb.log({"benchmark": asdict(benchmark_results)})
        
        logger.info(f"⚡ Benchmark: {samples_per_sec:.2f} samples/sec, {tokens_per_sec:.0f} tokens/sec, {memory_usage_gb:.2f}GB memory")
        
        return benchmark_results
    
    def create_training_dashboard(self, save_path: str = "./training_dashboard.png"):
        """Create a comprehensive training dashboard."""
        if not self.metrics_history:
            logger.warning("No metrics history available for dashboard")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Dashboard: {self.experiment_name}', fontsize=16)
        
        steps = [m.step for m in self.metrics_history]
        
        # Loss curves
        train_losses = [m.train_loss for m in self.metrics_history if m.train_loss is not None]
        eval_losses = [m.eval_loss for m in self.metrics_history if m.eval_loss is not None]
        
        axes[0, 0].plot(steps[:len(train_losses)], train_losses, label='Train Loss', color='blue')
        if eval_losses:
            eval_steps = [m.step for m in self.metrics_history if m.eval_loss is not None]
            axes[0, 0].plot(eval_steps, eval_losses, label='Eval Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        lrs = [m.learning_rate for m in self.metrics_history if m.learning_rate is not None]
        axes[0, 1].plot(steps[:len(lrs)], lrs, color='green')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # GPU Memory
        gpu_memory = [m.gpu_memory_gb for m in self.metrics_history if m.gpu_memory_gb is not None]
        if gpu_memory:
            axes[0, 2].plot(steps[:len(gpu_memory)], gpu_memory, color='purple')
            axes[0, 2].set_title('GPU Memory Usage')
            axes[0, 2].set_xlabel('Steps')
            axes[0, 2].set_ylabel('Memory (GB)')
            axes[0, 2].grid(True)
        
        # Training Speed
        speeds = [m.samples_per_second for m in self.metrics_history if m.samples_per_second is not None]
        if speeds:
            axes[1, 0].plot(steps[:len(speeds)], speeds, color='orange')
            axes[1, 0].set_title('Training Speed')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Samples/sec')
            axes[1, 0].grid(True)
        
        # CPU Usage
        cpu_usage = [m.cpu_percent for m in self.metrics_history if m.cpu_percent is not None]
        if cpu_usage:
            axes[1, 1].plot(steps[:len(cpu_usage)], cpu_usage, color='brown')
            axes[1, 1].set_title('CPU Usage')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('CPU %')
            axes[1, 1].grid(True)
        
        # Gradient Norm
        grad_norms = [m.grad_norm for m in self.metrics_history if m.grad_norm is not None and not np.isnan(m.grad_norm)]
        if grad_norms:
            axes[1, 2].plot(steps[:len(grad_norms)], grad_norms, color='red')
            axes[1, 2].set_title('Gradient Norm')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Grad Norm')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Training dashboard saved to: {save_path}")
        
        if self.enable_wandb:
            wandb.log({"training_dashboard": wandb.Image(save_path)})
        
        return save_path
    
    def save_experiment_summary(self, save_path: str = "./experiment_summary.json"):
        """Save comprehensive experiment summary."""
        summary = {
            "experiment_name": self.experiment_name,
            "total_runtime_seconds": time.time() - self.start_time,
            "total_steps": len(self.metrics_history),
            "metrics_count": len(self.metrics_history),
        }
        
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            summary.update({
                "final_train_loss": final_metrics.train_loss,
                "final_eval_loss": final_metrics.eval_loss,
                "final_learning_rate": final_metrics.learning_rate,
                "peak_gpu_memory_gb": max([m.gpu_memory_gb for m in self.metrics_history if m.gpu_memory_gb], default=0),
                "avg_training_speed": np.mean([m.samples_per_second for m in self.metrics_history if m.samples_per_second]),
            })
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 Experiment summary saved to: {save_path}")
        return summary
    
    def get_final_training_stats(self) -> Dict[str, Any]:
        """Calculate final training statistics including tokens per second."""
        if not self.metrics_history:
            return {}
        
        # Calculate overall tokens per second
        total_time = time.time() - self.start_time
        overall_tokens_per_sec = self.tokens_processed / total_time if total_time > 0 else 0
        
        # Get metrics with valid tokens per second
        valid_tokens_per_sec = [m.tokens_per_second for m in self.metrics_history if m.tokens_per_second is not None]
        avg_tokens_per_sec = np.mean(valid_tokens_per_sec) if valid_tokens_per_sec else 0
        
        # Get metrics with valid samples per second
        valid_samples_per_sec = [m.samples_per_second for m in self.metrics_history if m.samples_per_second is not None]
        avg_samples_per_sec = np.mean(valid_samples_per_sec) if valid_samples_per_sec else 0
        
        stats = {
            "total_training_time_seconds": total_time,
            "total_tokens_processed": self.tokens_processed,
            "overall_tokens_per_second": overall_tokens_per_sec,
            "average_tokens_per_second": avg_tokens_per_sec,
            "average_samples_per_second": avg_samples_per_sec,
            "total_steps": len(self.metrics_history),
            "final_loss": self.metrics_history[-1].train_loss if self.metrics_history else None,
        }
        
        return stats
    
    def finalize(self):
        """Finalize monitoring and cleanup."""
        logger.info("🎯 Finalizing performance monitoring...")
        
        # Get final stats
        final_stats = self.get_final_training_stats()
        
        # Log final training statistics with tokens per second
        logger.info("📈 Final Training Statistics:")
        logger.info(f"   Total training time: {final_stats.get('total_training_time_seconds', 0):.2f} seconds")
        logger.info(f"   Total tokens processed: {final_stats.get('total_tokens_processed', 0):,}")
        logger.info(f"   Overall tokens/second: {final_stats.get('overall_tokens_per_second', 0):.0f}")
        logger.info(f"   Average tokens/second: {final_stats.get('average_tokens_per_second', 0):.0f}")
        logger.info(f"   Average samples/second: {final_stats.get('average_samples_per_second', 0):.1f}")
        
        # Create final dashboard
        dashboard_path = self.create_training_dashboard()
        
        # Save experiment summary
        summary = self.save_experiment_summary()
        summary.update(final_stats)  # Add final stats to summary
        
        # Close W&B
        if self.enable_wandb:
            wandb.finish()
        
        logger.info(f"✅ Monitoring completed: {len(self.metrics_history)} metrics logged")
        return summary


def create_monitoring_config(
    project_name: str = "chemllm-training",
    experiment_name: Optional[str] = None,
    enable_wandb: bool = True,
    log_frequency: int = 10
) -> Dict[str, Any]:
    """Create performance monitoring configuration."""
    return {
        "project_name": project_name,
        "experiment_name": experiment_name,
        "enable_wandb": enable_wandb,
        "log_frequency": log_frequency,
    }


class TrainerWithMonitoring:
    """Wrapper to add monitoring to any Trainer class."""
    
    def __init__(self, trainer, monitor: PerformanceMonitor):
        self.trainer = trainer
        self.monitor = monitor
        self._original_log = trainer.log
        
        # Replace the log method with monitored version
        trainer.log = self._monitored_log
    
    def _monitored_log(self, logs: Dict[str, float], start_time=None):
        """Monitored logging that captures metrics."""
        # Get current step and epoch
        step = self.trainer.state.global_step
        epoch = self.trainer.state.epoch
        
        # Get batch information for accurate tokens per second calculation
        batch_info = None
        if hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader is not None:
            try:
                # Try to get actual batch size and sequence length from the trainer
                if hasattr(self.trainer.args, 'per_device_train_batch_size'):
                    batch_size = self.trainer.args.per_device_train_batch_size
                    # Estimate sequence length from the first batch if available
                    sequence_length = 256  # Default
                    try:
                        sample_batch = next(iter(self.trainer.train_dataloader))
                        if 'input_ids' in sample_batch:
                            sequence_length = sample_batch['input_ids'].shape[1]
                    except:
                        pass
                    
                    batch_info = {
                        "batch_size": batch_size,
                        "sequence_length": sequence_length
                    }
            except:
                pass
        
        # Log to monitor with batch information
        self.monitor.log_metrics(step, epoch, logs, batch_info)
        
        # Call original log method with proper arguments
        if start_time is not None:
            self._original_log(logs, start_time)
        else:
            self._original_log(logs)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original trainer."""
        return getattr(self.trainer, name)
