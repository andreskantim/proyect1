"""
Distributed training utilities for multi-GPU support.
Supports both DataParallel (single-node multi-GPU) and DDP (multi-node).
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
import os


class MultiGPUWrapper:
    """
    Wrapper to handle both single-GPU and multi-GPU training transparently.
    """

    def __init__(self, model: nn.Module, device: str = "cuda", strategy: str = "auto"):
        """
        Args:
            model: PyTorch model to wrap
            device: Device to use ('cuda', 'cpu', or specific like 'cuda:0')
            strategy: 'auto', 'single', 'dp' (DataParallel), or 'ddp' (DistributedDataParallel)
        """
        self.device = device
        self.num_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0

        # Determine strategy
        if strategy == "auto":
            if self.num_gpus > 1:
                self.strategy = "dp"  # DataParallel for multi-GPU single-node
            else:
                self.strategy = "single"
        else:
            self.strategy = strategy

        # Setup model
        self.model = self._setup_model(model)
        self.is_parallel = self.strategy in ["dp", "ddp"]

    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model based on strategy."""
        if self.strategy == "single":
            print(f"Using single GPU/CPU: {self.device}")
            return model.to(self.device)

        elif self.strategy == "dp":
            print(f"Using DataParallel with {self.num_gpus} GPUs")
            model = model.to(self.device)
            return nn.DataParallel(model)

        elif self.strategy == "ddp":
            # DDP setup (for future multi-node support)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            model = model.to(local_rank)
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def get_model(self) -> nn.Module:
        """Get the underlying model (unwrapped)."""
        if self.is_parallel:
            return self.model.module
        return self.model

    def get_wrapped_model(self) -> nn.Module:
        """Get the wrapped model (for training)."""
        return self.model

    def adjust_batch_size(self, base_batch_size: int) -> int:
        """
        Adjust batch size based on number of GPUs.
        DataParallel splits the batch across GPUs, so we scale it up.
        """
        if self.strategy == "dp":
            return base_batch_size * self.num_gpus
        return base_batch_size

    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint (handles unwrapping)."""
        state_dict = self.get_model().state_dict()
        torch.save({
            'model_state_dict': state_dict,
            **kwargs
        }, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint (handles unwrapping)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.get_model().load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def get_gpu_memory_info():
    """Get GPU memory usage for all available GPUs.

    Returns:
        dict: Dictionary with 'num_gpus' and 'devices' list, or
        str: "No GPUs available" if CUDA is not available
    """
    if not torch.cuda.is_available():
        return {"num_gpus": 0, "devices": []}

    num_gpus = torch.cuda.device_count()
    devices = []

    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        name = torch.cuda.get_device_name(i)

        devices.append({
            'id': i,
            'name': name,
            'memory_allocated_gb': allocated,
            'memory_reserved_gb': reserved,
            'memory_total_gb': total
        })

    return {
        'num_gpus': num_gpus,
        'devices': devices
    }


def setup_distributed(backend: str = "nccl"):
    """
    Initialize distributed training (for DDP).
    Call this at the start of your script if using multi-node training.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(local_rank)
        print(f"Initialized DDP: rank {rank}/{world_size}, local_rank {local_rank}")
        return True

    return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (for logging/saving)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
