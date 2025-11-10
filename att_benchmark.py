#!/usr/bin/env python3
"""
Benchmark script for scaled dot product attention at different scales.
Tests various combinations of head dimensions and sequence lengths.

Requirements:
    - PyTorch (pip install torch)
    - NumPy (pip install numpy)
    
Usage:
    python attention_benchmark.py
    
This script benchmarks the naive attention implementation to demonstrate
the O(seq_len²) memory complexity that causes OOM errors for long sequences.
FlashAttention addresses this issue by computing attention in tiles.
"""

import sys
import os
import time
import torch
import numpy as np
from itertools import product
import gc

import math

# Copy the necessary functions from model.py
def softmax(x, dim=-1):
    """Custom softmax implementation"""
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.
    
    This function implements Eq. 1 of the Transformer paper.
    
    Args:
        Q: Tensor of queries, shape (..., seq_len, d_k)
        K: Tensor of keys, shape (..., seq_len, d_k)
        V: Tensor of values, shape (..., seq_len, d_v)
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of False should
            be masked out.
    
    Returns:
        torch.FloatTensor of shape (..., seq_len, d_v)
    """
    d_k = K.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Shape: (..., seq_len, seq_len)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    # Apply softmax to get attention weights
    attention_weights = softmax(attention_scores, dim=-1)
    
    # Apply attention weights to values
    # Shape: (..., seq_len, d_v)
    return torch.matmul(attention_weights, V)


def create_causal_mask(seq_len, device):
    """Create a causal mask for attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def benchmark_attention(batch_size, d_k, seq_len, num_iterations=100, warmup_iterations=10, device='cuda'):
    """
    Benchmark the attention implementation for given parameters.
    
    Args:
        batch_size: Batch size (fixed at 8)
        d_k: Head embedding dimension
        seq_len: Sequence length
        num_iterations: Number of iterations to time
        warmup_iterations: Number of warmup iterations
        device: Device to run on
    
    Returns:
        dict: Contains forward_time, backward_time, peak_memory_mb
    """
    
    # Create random inputs - no head dimension as per requirements
    Q = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    
    # Create causal mask
    mask = create_causal_mask(seq_len, device)
    # Expand mask for batch dimension
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Warmup
    for _ in range(warmup_iterations):
        output = scaled_dot_product_attention(Q, K, V, mask)
        loss = output.sum()
        loss.backward()
        
        # Clear gradients
        Q.grad = None
        K.grad = None
        V.grad = None
        
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Clear GPU cache before measurement
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure forward pass
    forward_times = []
    for _ in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        output = scaled_dot_product_attention(Q, K, V, mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        forward_times.append(time.perf_counter() - start_time)
    
    # Measure memory before backward pass
    if device == 'cuda':
        torch.cuda.synchronize()
        memory_before_backward = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB
    else:
        memory_before_backward = 0  # CPU memory measurement not implemented
    
    # Measure backward pass
    backward_times = []
    for _ in range(num_iterations):
        # Forward pass (needed for backward)
        output = scaled_dot_product_attention(Q, K, V, mask)
        loss = output.sum()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        loss.backward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        backward_times.append(time.perf_counter() - start_time)
        
        # Clear gradients for next iteration
        Q.grad = None
        K.grad = None
        V.grad = None
    
    # Calculate statistics
    forward_time_ms = np.mean(forward_times) * 1000
    forward_std_ms = np.std(forward_times) * 1000
    backward_time_ms = np.mean(backward_times) * 1000
    backward_std_ms = np.std(backward_times) * 1000
    
    return {
        'forward_time_ms': forward_time_ms,
        'forward_std_ms': forward_std_ms,
        'backward_time_ms': backward_time_ms,
        'backward_std_ms': backward_std_ms,
        'memory_before_backward_mb': memory_before_backward
    }


def main():
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, running on CPU. Results may not be representative.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Fixed batch size
    batch_size = 8
    
    # Parameter ranges
    d_model_values = [16, 32, 64, 128]
    seq_len_values = [256, 1024, 4096, 8192, 16384]
    
    # Results storage
    results = []
    
    print(f"\nBenchmarking Scaled Dot Product Attention")
    print(f"Batch Size: {batch_size} (fixed)")
    print(f"Device: {device}")
    print(f"Iterations per measurement: 100")
    print("=" * 80)
    print(f"{'d_model':<10} {'seq_len':<10} {'Forward (ms)':<20} {'Backward (ms)':<20} {'Memory (MB)':<15}")
    print("-" * 80)
    
    # Iterate through all combinations
    for d_model, seq_len in product(d_model_values, seq_len_values):
        try:
            # Run benchmark
            metrics = benchmark_attention(
                batch_size=batch_size,
                d_k=d_model,
                seq_len=seq_len,
                num_iterations=100,
                warmup_iterations=10,
                device=device
            )
            
            # Store results
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                **metrics
            })
            
            # Print results
            print(f"{d_model:<10} {seq_len:<10} "
                  f"{metrics['forward_time_ms']:.3f}±{metrics['forward_std_ms']:.3f}      "
                  f"{metrics['backward_time_ms']:.3f}±{metrics['backward_std_ms']:.3f}      "
                  f"{metrics['memory_before_backward_mb']:.2f}")
            
            # Clear GPU cache between runs
            if device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{d_model:<10} {seq_len:<10} {'OOM':<20} {'OOM':<20} {'OOM':<15}")
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'forward_time_ms': float('inf'),
                'forward_std_ms': float('inf'),
                'backward_time_ms': float('inf'),
                'backward_std_ms': float('inf'),
                'memory_before_backward_mb': float('inf'),
                'error': 'OOM'
            })
            
            # Clear GPU memory after OOM
            if device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"{d_model:<10} {seq_len:<10} Error: {str(e)}")
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    
    # Analyze scaling behavior
    print("\n" + "=" * 80)
    print("Analysis:")
    print("-" * 80)
    
    # Find configurations that ran successfully
    successful_runs = [r for r in results if 'error' not in r]
    if successful_runs:
        # Find fastest and slowest configurations
        fastest_forward = min(successful_runs, key=lambda x: x['forward_time_ms'])
        slowest_forward = max(successful_runs, key=lambda x: x['forward_time_ms'])
        
        print(f"Fastest forward pass: d_model={fastest_forward['d_model']}, "
              f"seq_len={fastest_forward['seq_len']}, "
              f"time={fastest_forward['forward_time_ms']:.3f}ms")
        print(f"Slowest forward pass: d_model={slowest_forward['d_model']}, "
              f"seq_len={slowest_forward['seq_len']}, "
              f"time={slowest_forward['forward_time_ms']:.3f}ms")
        
        # Memory usage analysis
        max_memory_run = max(successful_runs, key=lambda x: x['memory_before_backward_mb'])
        print(f"Peak memory usage: d_model={max_memory_run['d_model']}, "
              f"seq_len={max_memory_run['seq_len']}, "
              f"memory={max_memory_run['memory_before_backward_mb']:.2f}MB")
        
        # Count OOM errors
        oom_count = len([r for r in results if r.get('error') == 'OOM'])
        if oom_count > 0:
            print(f"\nOut of Memory errors: {oom_count}/{len(results)} configurations")
            print("Note: The naive attention implementation has O(seq_len²) memory complexity,")
            print("which causes OOM errors for long sequences. FlashAttention addresses this issue.")
    
    # Save results to file
    import json
    with open('attention_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to attention_benchmark_results.json")


if __name__ == "__main__":
    main()