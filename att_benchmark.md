## benchmarking_script

uv run python benchmark.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --bm_mode "forward_backward"

[forward_backward] B=8, T=1024, V=50257, L=12, d=768, h=12 on CPU — 606.90±3.13 ms per step (warmup=5, measure=10)
[forward] B=8, T=1024, V=50257, L=12, d=768, h=12 on CPU — 215.24±1.30 ms per step (warmup=5, measure=10)

uv run python benchmark.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --bm_mode "forward_backward"

*OOM*


## mixed_precision
uv run python benchmark.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --bm_mode "forward" --bm_warmup_steps 5 --use_mixed_precision

[forward_backward] B=8, T=1024, V=50257, L=12, d=768, h=12 on CPU — 385.71±0.31 ms per step (warmup=5, measure=10)
[forward] B=8, T=1024, V=50257, L=12, d=768, h=12 on CPU — 124.47±0.06 ms per step (warmup=5, measure=10)

uv run python benchmark.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --bm_mode "forward_backward" --bm_warmup_steps 5 --use_mixed_precision

[forward_backward] B=8, T=1024, V=50257, L=24, d=1024, h=16 on CPU — 961.94±0.90 ms per step (warmup=5, measure=10)
[forward] B=8, T=1024, V=50257, L=24, d=1024, h=16 on CPU — 306.09±0.22 ms per step (warmup=5, measure=10)

uv run python benchmark.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --bm_mode "forward_backward" --bm_warmup_steps 5 --use_mixed_precision

*OOM*


## attention_benchmarking
uv run python  att_benchmark.py

Using GPU: NVIDIA RTX 6000 Ada Generation

Benchmarking Scaled Dot Product Attention
Batch Size: 8 (fixed)
Device: cuda
Iterations per measurement: 100
================================================================================
d_model    seq_len    Forward (ms)         Backward (ms)        Memory (MB)
--------------------------------------------------------------------------------
16         256        0.282±0.059      0.661±0.062      28.86
16         1024       0.369±0.091      1.032±0.071      211.44
16         4096       9.139±0.377      21.141±0.102      3113.00
16         8192       36.254±0.685      83.602±0.091      12385.75
16         16384      OOM                  OOM                  OOM
32         256        0.179±0.007      0.492±0.094      29.36
32         1024       0.426±0.419      0.948±0.054      213.44
32         4096       9.279±0.426      21.174±0.057      3121.00
32         8192       36.430±0.660      83.943±0.122      12401.75
32         16384      OOM                  OOM                  OOM
64         256        0.184±0.026      0.464±0.110      30.36
64         1024       0.439±0.417      0.962±0.034      217.44
64         4096       9.402±0.414      21.353±0.056      3137.00
64         8192       36.902±0.623      84.647±0.084      12433.75
64         16384      OOM                  OOM                  OOM
128        256        0.190±0.033      0.488±0.168      32.36
128        1024       0.469±0.387      1.007±0.023      225.44
128        4096       9.856±0.362      21.901±0.057      3169.00
128        8192       39.049±0.597      86.983±0.235      12497.75
128        16384      OOM                  OOM                  OOM

================================================================================
Benchmark Complete!

================================================================================
Analysis:
--------------------------------------------------------------------------------
Fastest forward pass: d_model=32, seq_len=256, time=0.179ms
Slowest forward pass: d_model=128, seq_len=8192, time=39.049ms
Peak memory usage: d_model=128, seq_len=8192, memory=12497.75MB

Out of Memory errors: 4/20 configurations
Note: The naive attention implementation has O(seq_len²) memory complexity,


## attention_benchmarking_compiled
uv run python  compiled_benchmark.py

================================================================================
COMPILED vs UNCOMPILED PYTORCH BENCHMARKING
================================================================================
PyTorch Version: 2.6.0+cu124
Device: cuda
GPU: NVIDIA RTX 6000 Ada Generation

================================================================================
PART (a): ATTENTION BENCHMARKING - Compiled vs Uncompiled
================================================================================
Device: cuda

Batch Size: 8

----------------------------------------------------------------------------------------------------
d_model  seq_len  | Uncompiled (ms)                | Compiled (ms)                  | Speedup
----------------------------------------------------------------------------------------------------
16       256      | F:  0.48 B:  1.14 | F:  0.40 B:  0.68 | F:1.21x B:1.69x
16       1024     | F:  0.38 B:  1.17 | F:  0.27 B:  0.61 | F:1.39x B:1.90x
16       4096     | F:  9.09 B: 21.11 | F:  2.99 B:  8.27 | F:3.04x B:2.55x
16       8192     | F: 36.11 B: 83.69 | F: 12.80 B: 33.13 | F:2.82x B:2.53x
16       16384    | OOM
32       256      | F:  0.23 B:  0.52 | F:  0.26 B:  0.75 | F:0.89x B:0.70x
32       1024     | F:  0.37 B:  1.15 | F:  0.29 B:  0.64 | F:1.28x B:1.81x
32       4096     | F:  9.12 B: 21.23 | F:  3.61 B:  8.45 | F:2.53x B:2.51x
32       8192     | F: 36.31 B: 83.98 | F: 16.74 B: 35.22 | F:2.17x B:2.38x
32       16384    | OOM
64       256      | F:  0.22 B:  1.14 | F:  0.20 B:  0.73 | F:1.10x B:1.56x
64       1024     | F:  0.38 B:  1.17 | F:  0.29 B:  0.65 | F:1.32x B:1.78x
64       4096     | F:  9.25 B: 21.39 | F:  4.00 B:  8.86 | F:2.31x B:2.41x
64       8192     | F: 36.78 B: 84.77 | F: 18.29 B: 37.00 | F:2.01x B:2.29x
64       16384    | OOM
128      256      | F:  0.23 B:  0.49 | F:  0.20 B:  0.34 | F:1.12x B:1.41x
128      1024     | F:  0.44 B:  1.02 | F:  0.33 B:  0.72 | F:1.31x B:1.43x
128      4096     | F:  9.77 B: 21.98 | F:  4.76 B:  9.85 | F:2.05x B:2.23x
128      8192     | F: 38.99 B: 87.05 | F: 21.70 B: 40.16 | F:1.80x B:2.17x
128      16384    | OOM
----------------------------------------------------------------------------------------------------
Average Speedup - Forward: 1.77x, Backward: 1.96x

================================================================================
PART (b): FULL TRANSFORMER BENCHMARKING - Compiled vs Uncompiled
================================================================================
Device: cuda

------------------------------------------------------------------------------------------------------------------------
Config (B-T-d-L-H)   | Vanilla (ms)                        | Compiled (ms)                       | Speedup
------------------------------------------------------------------------------------------------------------------------
4-256-256-4-8        | F:   9.2 FB:  20.9 S:  25.3 | F:   2.4 FB:  12.2 S:  16.3 | F:3.88x FB:1.71x S:1.55x
4-512-512-6-8        | F:  14.6 FB:  49.3 S:  59.9 | F:  10.0 FB:  42.2 S:  50.5 | F:1.46x FB:1.17x S:1.19x
8-256-768-6-12       | F:  18.4 FB:  62.6 S:  76.7 | F:  16.8 FB:  57.9 S:  70.2 | F:1.09x FB:1.08x S:1.09x
8-512-768-12-12      | F:  83.1 FB: 249.5 S: 267.5 | F:  70.0 FB: 206.0 S: 224.3 | F:1.19x FB:1.21x S:1.19x
------------------------------------------------------------------------------------------------------------------------
Average Speedups - Forward: 1.90x, Forward+Backward: 1.29x, Full Step: 1.26x

================================================================================
BENCHMARKING COMPLETE
================================================================================
Results saved to: benchmark_results_clean.json
