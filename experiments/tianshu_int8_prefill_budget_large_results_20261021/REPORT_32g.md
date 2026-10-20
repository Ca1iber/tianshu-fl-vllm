# Larger scheduling budgets: GUP INT8 component gate, BI-V150 32G, 2026-10-21

Component numeric gate passes; not yet a model or scheduler throughput result. M8192/16384 here represents several 4K requests combined, not a full 16K request benchmark.

Actual layer-0 GUP weights [12288,2048] were tested at M4096/8192/16384 with two preselected configurations of the frozen grouped NN kernel. All six cases compiled and passed three varied GPU reference phases each. Independent CPU verification recomputed 18 selected INT64 products and all 176,160,768 quantized activation values; full-output numeric comparisons run on the GPU, not on the CPU. Seven alternating rounds of 20 graph replays include dynamic activation quantization and exclude static weight packing.

| M | BF16 linear ms | selected A8W8 ms | throughput gain |
|---:|---:|---:|---:|
| 4096 | 2.364367 | 2.008850 | +17.70% |
| 8192 | 4.729503 | 3.980265 | +18.82% |
| 16384 | 9.461415 | 7.901703 | +19.74% |

All selected tiles use BM128 BN512 BK64, eight warps, one stage, group_m=8, with 256 registers and zero spills. The same frozen kernel is used at all three sizes. The qualified model currently uses budget4096 and rejects larger shapes; a separate explicit policy fork, extended startup warming, actual memory accounting and full-model A/B are required before adopting a larger budget. Existing mixed-prefill routing already stages only prefill sequences, so no duplicate mixed-split optimization is claimed.

Full raw results and tensors remain at /workspace/benchmark-results/20261021_int8_prefill_budget_large_GPU_tianshu. This directory retains the independent verifier, numeric gate and key timing JSON/CSV.
