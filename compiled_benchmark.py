import sys
import os
import time
import torch
import numpy as np
from itertools import product
import gc
import json
import warnings

from cs336_basics.model import BasicsTransformerLM, scaled_dot_product_attention
from cs336_basics.nn_utils import softmax, log_softmax, cross_entropy
from cs336_basics.optimizer import AdamW

warnings.filterwarnings('ignore', category=UserWarning)


def detect_device():
    """Detect the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def create_causal_mask(seq_len, device):
    """Create a causal mask for attention"""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def benchmark_attention(attention_fn, batch_size, d_k, seq_len, 
                        num_iterations=100, warmup_iterations=10, device='cuda'):
    """Benchmark an attention function"""
    
    # Create inputs
    Q = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)  
    V = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
    mask = create_causal_mask(seq_len, device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Warmup
    for _ in range(warmup_iterations):
        output = attention_fn(Q, K, V, mask)
        loss = output.sum()
        loss.backward()
        Q.grad, K.grad, V.grad = None, None, None
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
    
    # Time forward passes
    forward_times = []
    for _ in range(num_iterations):
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        start = time.perf_counter()
        output = attention_fn(Q, K, V, mask)
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        forward_times.append(time.perf_counter() - start)
    
    # Time backward passes
    backward_times = []
    for _ in range(num_iterations):
        output = attention_fn(Q, K, V, mask)
        loss = output.sum()
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        start = time.perf_counter()
        loss.backward()
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        backward_times.append(time.perf_counter() - start)
        Q.grad, K.grad, V.grad = None, None, None
    
    return {
        'forward_ms': np.mean(forward_times) * 1000,
        'forward_std': np.std(forward_times) * 1000,
        'backward_ms': np.mean(backward_times) * 1000,
        'backward_std': np.std(backward_times) * 1000,
    }


def benchmark_transformer(model, batch_size, context_length, vocab_size,
                         num_iterations=10, warmup_iterations=5, device='cuda'):
    """Benchmark a transformer model"""
    
    # Create data
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Warmup
    for _ in range(warmup_iterations):
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
    
    # Time forward pass
    forward_times = []
    for _ in range(num_iterations):
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        forward_times.append(time.perf_counter() - start)
    
    # Time forward + backward
    fwd_bwd_times = []
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        start = time.perf_counter()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        fwd_bwd_times.append(time.perf_counter() - start)
    
    # Time full training step
    full_step_times = []
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        start = time.perf_counter()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        if device in ['cuda', 'mps']:
            torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize()
        
        full_step_times.append(time.perf_counter() - start)
    
    return {
        'forward_ms': np.mean(forward_times) * 1000,
        'fwd_bwd_ms': np.mean(fwd_bwd_times) * 1000,
        'full_step_ms': np.mean(full_step_times) * 1000,
    }


def part_a_attention_benchmark():
    """Part (a): Compare compiled vs uncompiled attention"""
    print("\n" + "="*80)
    print("PART (a): ATTENTION BENCHMARKING - Compiled vs Uncompiled")
    print("="*80)
    
    device = detect_device()
    print(f"Device: {device}")
    
    # Check if compilation makes sense for this device
    if device not in ['cuda']:
        print(f"\n⚠️  Warning: torch.compile may not improve performance on {device}")
    
    batch_size = 8
    d_model_values = [16, 32, 64, 128]
    seq_len_values = [256, 1024, 4096, 8192, 16384]
    
    # Compile the attention function
    compiled_attention = torch.compile(scaled_dot_product_attention)
    
    print(f"\nBatch Size: {batch_size}")
    print("\n" + "-"*100)
    print(f"{'d_model':<8} {'seq_len':<8} | {'Uncompiled (ms)':<30} | {'Compiled (ms)':<30} | {'Speedup':<10}")
    print("-"*100)
    
    results = []
    for d_model, seq_len in product(d_model_values, seq_len_values):
        try:
            # Benchmark uncompiled
            uncomp = benchmark_attention(
                scaled_dot_product_attention,
                batch_size, d_model, seq_len, device=device
            )
            
            # Clear cache
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Benchmark compiled
            comp = benchmark_attention(
                compiled_attention,
                batch_size, d_model, seq_len, device=device
            )
            
            # Calculate speedups
            fwd_speedup = uncomp['forward_ms'] / comp['forward_ms']
            bwd_speedup = uncomp['backward_ms'] / comp['backward_ms']
            
            print(f"{d_model:<8} {seq_len:<8} | "
                  f"F:{uncomp['forward_ms']:6.2f} B:{uncomp['backward_ms']:6.2f} | "
                  f"F:{comp['forward_ms']:6.2f} B:{comp['backward_ms']:6.2f} | "
                  f"F:{fwd_speedup:.2f}x B:{bwd_speedup:.2f}x")
            
            results.append({
                'd_model': d_model, 'seq_len': seq_len,
                'uncompiled': uncomp, 'compiled': comp,
                'fwd_speedup': fwd_speedup, 'bwd_speedup': bwd_speedup
            })
            
            # Clear cache
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{d_model:<8} {seq_len:<8} | OOM")
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    print("-"*100)
    
    if results:
        avg_fwd = np.mean([r['fwd_speedup'] for r in results])
        avg_bwd = np.mean([r['bwd_speedup'] for r in results])
        print(f"Average Speedup - Forward: {avg_fwd:.2f}x, Backward: {avg_bwd:.2f}x")
    
    return results


def part_b_transformer_benchmark():
    """Part (b): Compare compiled vs uncompiled full transformer"""
    print("\n" + "="*80)
    print("PART (b): FULL TRANSFORMER BENCHMARKING - Compiled vs Uncompiled")
    print("="*80)
    
    device = detect_device()
    print(f"Device: {device}")
    
    configs = [
        {'batch_size': 4, 'context_length': 256, 'd_model': 256, 'num_layers': 4, 'num_heads': 8, 'd_ff': 1024},
        {'batch_size': 4, 'context_length': 512, 'd_model': 512, 'num_layers': 6, 'num_heads': 8, 'd_ff': 2048},
        {'batch_size': 8, 'context_length': 256, 'd_model': 768, 'num_layers': 6, 'num_heads': 12, 'd_ff': 2048},
        {'batch_size': 8, 'context_length': 512, 'd_model': 768, 'num_layers': 12, 'num_heads': 12, 'd_ff': 3072},
    ]
    
    vocab_size = 50257
    rope_theta = 10000.0
    
    print("\n" + "-"*120)
    print(f"{'Config (B-T-d-L-H)':<20} | {'Vanilla (ms)':<35} | {'Compiled (ms)':<35} | {'Speedup':<20}")
    print("-"*120)
    
    results = []
    for cfg in configs:
        config_str = f"{cfg['batch_size']}-{cfg['context_length']}-{cfg['d_model']}-{cfg['num_layers']}-{cfg['num_heads']}"
        
        try:
            # Create and benchmark vanilla model
            model_vanilla = BasicsTransformerLM(
                vocab_size=vocab_size,
                context_length=cfg['context_length'],
                d_model=cfg['d_model'],
                num_layers=cfg['num_layers'],
                num_heads=cfg['num_heads'],
                d_ff=cfg['d_ff'],
                rope_theta=rope_theta
            ).to(device)
            
            vanilla = benchmark_transformer(
                model_vanilla, cfg['batch_size'], cfg['context_length'], vocab_size, device=device
            )
            
            # Clean up
            del model_vanilla
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Create and benchmark compiled model
            model_compiled = BasicsTransformerLM(
                vocab_size=vocab_size,
                context_length=cfg['context_length'],
                d_model=cfg['d_model'],
                num_layers=cfg['num_layers'],
                num_heads=cfg['num_heads'],
                d_ff=cfg['d_ff'],
                rope_theta=rope_theta
            ).to(device)
            
            model_compiled = torch.compile(model_compiled)
            
            compiled = benchmark_transformer(
                model_compiled, cfg['batch_size'], cfg['context_length'], vocab_size, device=device
            )
            
            # Calculate speedups
            fwd_speedup = vanilla['forward_ms'] / compiled['forward_ms']
            fwd_bwd_speedup = vanilla['fwd_bwd_ms'] / compiled['fwd_bwd_ms']
            full_speedup = vanilla['full_step_ms'] / compiled['full_step_ms']
            
            print(f"{config_str:<20} | "
                  f"F:{vanilla['forward_ms']:6.1f} FB:{vanilla['fwd_bwd_ms']:6.1f} S:{vanilla['full_step_ms']:6.1f} | "
                  f"F:{compiled['forward_ms']:6.1f} FB:{compiled['fwd_bwd_ms']:6.1f} S:{compiled['full_step_ms']:6.1f} | "
                  f"F:{fwd_speedup:.2f}x FB:{fwd_bwd_speedup:.2f}x S:{full_speedup:.2f}x")
            
            results.append({
                'config': cfg,
                'vanilla': vanilla,
                'compiled': compiled,
                'speedups': {'forward': fwd_speedup, 'fwd_bwd': fwd_bwd_speedup, 'full': full_speedup}
            })
            
            # Clean up
            del model_compiled
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"{config_str:<20} | Error: {str(e)}")
    
    print("-"*120)
    
    if results:
        avg_fwd = np.mean([r['speedups']['forward'] for r in results])
        avg_fwd_bwd = np.mean([r['speedups']['fwd_bwd'] for r in results])
        avg_full = np.mean([r['speedups']['full'] for r in results])
        print(f"Average Speedups - Forward: {avg_fwd:.2f}x, Forward+Backward: {avg_fwd_bwd:.2f}x, Full Step: {avg_full:.2f}x")
    
    return results


def main():
    print("="*80)
    print("COMPILED vs UNCOMPILED PYTORCH BENCHMARKING")
    print("="*80)
    print(f"PyTorch Version: {torch.__version__}")
    
    device = detect_device()
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print("Apple Silicon GPU (MPS)")
        print("⚠️  Note: torch.compile has limited support on MPS and may be slower")
    else:
        print("CPU - torch.compile may not provide speedup")
    
    # Part (a): Attention benchmarking
    attention_results = part_a_attention_benchmark()
    
    # Part (b): Full transformer benchmarking
    transformer_results = part_b_transformer_benchmark()
    
    # Save results
    results = {
        'device': device,
        'attention': attention_results,
        'transformer': transformer_results
    }
    
    with open('benchmark_results_clean.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE")
    print("="*80)
    print("Results saved to: benchmark_results_clean.json")
    
    if device in ['mps', 'cpu']:
        print(f"\n⚠️  Note: torch.compile typically doesn't improve performance on {device}")
        print("For best results, use CUDA GPUs or consider device-specific optimizations")


if __name__ == "__main__":
    main()