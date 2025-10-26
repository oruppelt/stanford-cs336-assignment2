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

    parser.add_argument("--bm_mode", type=str, default="forward", choices=["forward", "forward_backward"],
                        help="Benchmark only forward, or forward+backward")
    parser.add_argument("--bm_warmup_steps", type=int, default=5, help="Warm-up steps (not timed)")
    parser.add_argument("--bm_measure_steps", type=int, default=10, help="Measured steps")

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

@torch.no_grad
def forward_pass(model, x, y):
    logits = model.forward(x)
    loss = cross_entropy(logits, y)

    return float(loss)

def forward_backward(model, x, y):
    model.zero_grad()

    logits = model.forward(x)
    loss = cross_entropy(logits, y)

    loss.backward()

    return loss

def make_batch(batch, context, vocab, device):
    x = torch.randint(0, vocab, (batch, context), device = device)
    y = torch.randint(0, vocab, (batch, context), device = device)

    return x, y

def synchronize_if_cuda(device: torch.device):
    if device == "cuda":
        torch.cuda.synchronize()

def benchmark(model, x, y, mode: str, warmup_steps: int, measure_steps: int, device: torch.device):
    # Warm-up (not timed)
    if mode == "forward":
        # nvtx.range_push("warmup")
        for _ in range(warmup_steps):
            forward_pass(model, x, y)
            synchronize_if_cuda(device)
    else:
        for _ in range(warmup_steps):
            forward_backward(model, x, y)
            synchronize_if_cuda(device)

    # Measured loop
    times = []
    losses = []
    for _ in range(measure_steps):
        start = timeit.default_timer()
        if mode == "forward":
            loss = forward_pass(model, x, y)
        else:
            loss = forward_backward(model, x, y)
        synchronize_if_cuda(device)
        elapsed = timeit.default_timer() - start
        times.append(elapsed)
        losses.append(loss)

    return times, losses

def summarize(times_s):
    n = len(times_s)
    mean_s = sum(times_s) / n
    var = sum((t - mean_s) ** 2 for t in times_s) / n
    std_s = math.sqrt(var)
    return mean_s * 1e3, std_s * 1e3 


def main():
    # Parse minimal args
    args = parse_arguments()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    print(f"Using device: {device}")

    model = build_model(args, device)

    model.train()  # keep model in train mode by default; no loops will run here
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready. Parameters: {total_params:,}")
    print("Pass complete (no training, no benchmarking yet).")

    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device)

    times, _ = benchmark(
        model=model,
        x=x,
        y=y,
        mode=args.bm_mode,
        warmup_steps=args.bm_warmup_steps,
        measure_steps=args.bm_measure_steps,
        device=device,
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

