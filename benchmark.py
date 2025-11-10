import sys
import os
sys.path.append('cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax, log_softmax, cross_entropy

import argparse
import math
import timeit
import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda import nvtx
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext

def parse_arguments():

    parser = argparse.ArgumentParser(description="Pruned Transformer setup (removal-only)")

    # Minimal, model-related config
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length (sequence length)")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Batch size kept for later (when adding benchmarking), does nothing yet
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (unused in this pruned file)")

    # Device / reproducibility
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Mixed precision settings
    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Enable mixed precision training with BF16")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="Precision type for mixed precision (bf16 recommended)")

    parser.add_argument("--bm_mode", type=str, default="forward", choices=["forward", "forward_backward"],
                        help="Benchmark only forward, or forward+backward")
    parser.add_argument("--bm_warmup_steps", type=int, default=5, help="Warm-up steps (not timed)")
    parser.add_argument("--bm_measure_steps", type=int, default=10, help="Measured steps")

    # Optimizer parameters for full_step mode
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")

    return parser.parse_args()


def select_device(requested: str) -> str:

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    return requested


def set_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def build_model(args, device: torch.device):
    print("Initializing model...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        d_ff=args.d_ff
    ).to(device)

    return model

def check_bf16_support(device: torch.device):
    """Check if the current device supports BF16"""
    if device.type != "cuda":
        return False

    # Check compute capability for BF16 support (>= 8.0 for Ampere and newer)
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10

    if compute_capability < 8.0:
        print(f"Warning: GPU compute capability {compute_capability:.1f} may not fully support BF16.")
        print("BF16 is optimally supported on Ampere (compute capability 8.0) and newer GPUs.")
        return False

    return True

def get_autocast_context(args, device):
    """Get the appropriate autocast context based on settings"""
    if not args.use_mixed_precision:
        return nullcontext()

    if device.type == "cuda":
        dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    elif device.type == "cpu":
        # CPU autocast support for BF16
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        print(f"Warning: Mixed precision not supported for device type {device.type}")
        return nullcontext()

class Nvtx:
    enabled = torch.cuda.is_available()

    @staticmethod
    def push(name: str):
        if Nvtx.enabled:
            # from torch.cuda import nvtx
            nvtx.range_push(name)

    @staticmethod
    def pop():
        if Nvtx.enabled:
            # from torch.cuda import nvtx
            nvtx.range_pop()

@torch.no_grad
def forward_pass(model, x, y, autocast_context):
    Nvtx.push("forward_pass")

    # Annotate model forward
    Nvtx.push("model_forward")
    with autocast_context:
        logits = model.forward(x)
    Nvtx.pop()

    # Annotate loss computation
    Nvtx.push("loss_computation")
    with autocast_context:
        loss = cross_entropy(logits, y)
    Nvtx.pop()  # loss_computation

    Nvtx.pop()  # forward_pass
    return float(loss)

def forward_backward(model, x, y, autocast_context, grad_scaler=None):
    Nvtx.push("forward_backward")

    # Zero gradients
    Nvtx.push("zero_grad")
    model.zero_grad()
    Nvtx.pop()  # zero_grad

    # Forward pass
    Nvtx.push("forward")
    Nvtx.push("model_forward")
    with autocast_context:
        logits = model.forward(x)
    Nvtx.pop()  # model_forward

    Nvtx.push("loss_computation")
    with autocast_context:
        loss = cross_entropy(logits, y)
    Nvtx.pop()  # loss_computation
    Nvtx.pop()  # forward

    # Backward pass
    Nvtx.push("backward")
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()
    Nvtx.pop()  # backward

    Nvtx.pop()  # forward_backward
    return loss

def full_training_step(model, optimizer, x, y, autocast_context, grad_scaler=None):
    Nvtx.push("full_training_step")

    # Zero gradients
    Nvtx.push("zero_grad")
    optimizer.zero_grad()
    Nvtx.pop()  # zero_grad

    # Forward pass
    Nvtx.push("forward")
    Nvtx.push("model_forward")
    with autocast_context:
        logits = model.forward(x)
    Nvtx.pop()  # model_forward

    Nvtx.push("loss_computation")
    with autocast_context:
        loss = cross_entropy(logits, y)
    Nvtx.pop()  # loss_computation
    Nvtx.pop()  # forward

    # Backward pass
    Nvtx.push("backward")
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()
    Nvtx.pop()  # backward

    # Optimizer step
    Nvtx.push("optimizer_step")
    if grad_scaler is not None:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    Nvtx.pop()  # optimizer_step

    Nvtx.pop()  # full_training_step
    return loss

def make_batch(batch, context, vocab, device):
    x = torch.randint(0, vocab, (batch, context), device=device)
    y = torch.randint(0, vocab, (batch, context), device=device)

    return x, y

def synchronize_if_cuda(device: torch.device):
    if device == "cuda":
        torch.cuda.synchronize()

def benchmark(model, x, y, mode: str, warmup_steps: int, measure_steps: int, device: torch.device, autocast_context,
              optimizer=None, grad_scaler=None):
    # Warm-up (not timed)
    Nvtx.push("warmup")

    if mode == "forward":
        for _ in range(warmup_steps):
            Nvtx.push("warmup_step")
            forward_pass(model, x, y, autocast_context)
            synchronize_if_cuda(device)
            Nvtx.pop()  # warmup_step
    elif mode == "forward_backward":
        for _ in range(warmup_steps):
            Nvtx.push("warmup_step")
            forward_backward(model, x, y, autocast_context, grad_scaler)
            synchronize_if_cuda(device)
            Nvtx.pop()
    else:
        if optimizer is None:
            raise ValueError("Optimizer required for full_step mode")
        for i in range(warmup_steps):
            Nvtx.push(f"warmup_step_{i}")
            full_training_step(model, optimizer, x, y, autocast_context, grad_scaler)
            synchronize_if_cuda(device)
            Nvtx.pop()  # warmup_step_i

    Nvtx.pop()  # warmup

    # Measured loop
    Nvtx.push("measured")

    times = []
    losses = []
    for i in range(measure_steps):
        Nvtx.push(f"measured_step_{i}")

        start = timeit.default_timer()
        if mode == "forward":
            loss = forward_pass(model, x, y, autocast_context)
        elif mode == "forward_backward":
            loss = forward_backward(model, x, y, autocast_context, grad_scaler)
        else:  # full_step
            loss = full_training_step(model, optimizer, x, y, autocast_context, grad_scaler)
        synchronize_if_cuda(device)
        elapsed = timeit.default_timer() - start

        Nvtx.pop()  # measure_step_i

        times.append(elapsed)
        losses.append(loss)

    Nvtx.pop()  # measured

    return times, losses

def summarize(times_s):
    n = len(times_s)
    mean_s = sum(times_s) / n
    var = sum((t - mean_s) ** 2 for t in times_s) / n
    std_s = math.sqrt(var)
    return mean_s * 1e3, std_s * 1e3

def print_precision_info(args, device):
    """Print information about precision settings"""
    if args.use_mixed_precision:
        precision_str = args.precision.upper()
        print(f"Mixed Precision: ENABLED ({precision_str})")

        if args.precision == "bf16":
            if device.type == "cuda":
                has_bf16 = check_bf16_support(device)
                if has_bf16:
                    print(f"BF16 Support: ✓ (Compute capability {torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]})")
                else:
                    print("BF16 Support: Limited (may run with reduced performance)")
            print("Note: BF16 does not require GradScaler")
        else:  # fp16
            print("Note: Using FP16 with GradScaler for gradient scaling")
    else:
        print("Mixed Precision: DISABLED (using FP32)")


def main():

    print(f"Before empty_cache GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set memory allocation config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    args = parse_arguments()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    print(f"Using device: {device}")
    device = torch.device(device)

    print_precision_info(args, device)

    autocast_context = get_autocast_context(args, device)

    grad_scaler = None
    if args.use_mixed_precision and args.precision == "fp16" and device.type == "cuda":
        grad_scaler = torch.GradScaler('cuda')
        print("Initialized GradScaler for FP16 mixed precision")

    model = build_model(args, device)

    model.train()

    # Create optimizer if needed for full_step mode
    optimizer = None
    if args.bm_mode == "full_step":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print(f"Created AdamW optimizer with lr={args.learning_rate}, weight_decay={args.weight_decay}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready. Parameters: {total_params:,}")

    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device)

    print(f"Before benchmark GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    times, _ = benchmark(
        model=model,
        x=x,
        y=y,
        mode=args.bm_mode,
        warmup_steps=args.bm_warmup_steps,
        measure_steps=args.bm_measure_steps,
        device=device,
        optimizer=optimizer,
        autocast_context=autocast_context,
        grad_scaler=grad_scaler
    )

    mean_ms, std_ms = summarize(times)

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(
        f"[{args.bm_mode}] "
        f"B={args.batch_size}, T={args.context_length}, V={args.vocab_size}, "
        f"L={args.num_layers}, d={args.d_model}, h={args.num_heads} "
        f"on {gpu} — {mean_ms:.2f}±{std_ms:.2f} ms per step "
        f"(warmup={args.bm_warmup_steps}, measure={args.bm_measure_steps})"
    )

if __name__ == "__main__":
    main()
