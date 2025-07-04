"""
Core interfaces package for the refactored training pipeline.

This package defines the contracts that all implementations must follow,
enabling dependency injection and modular architecture.
"""

__version__ = "1.0.0"
__all__ = [
    "IDataSource",
    "IModel", 
    "IOptimizer",
    "ITrainer",
    "IEvaluator", 
    "ICheckpointer",
    "IMonitor"
]
