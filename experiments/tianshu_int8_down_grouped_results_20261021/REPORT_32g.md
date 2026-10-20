# Down INT8 / BF16 linear component exploration, BI-V150 32G, 2026-10-21

Not adopted. No model accuracy or end-to-end throughput claim follows from these component measurements.

The actual layer-0 Down weight [2048,6144] was tested at M=8,64,128,129,256,512,1024,2048,4096, with 20 configurations each. All 180 configurations compiled. GPU checks covered three changed-input/weight/scale phases per configuration. The separate CPU verifier checked 540 selected INT64 matrix products and all 152,340,480 quantized activation values. GPU full-output checks compare against the quantized reference; CPU checks recompute selected outputs, not every GPU output. Seven rotating rounds of 20 graph replays include dynamic activation quantization; static checkpoint packing is excluded.

The baseline is actual Down BF16 mm, retaining the existing M2048 special tile. An auxiliary BF16 linear implementation uses the same original weight and input. The generated native model calls extern_kernels.mm(..., out=...), as shown in NATIVE_COMPILE_EXCERPT_32g.txt.

| M | BF16 mm ms | best A8W8 ms | throughput change |
|---:|---:|---:|---:|
| 8 | 0.092385 | 0.115770 | -20.20% |
| 64 | 0.059774 | 0.079177 | -24.51% |
| 128 | 0.090483 | 0.081269 | +11.34% |
| 129 | 0.106524 | 0.211227 | -49.57% |
| 256 | 0.176882 | 0.099956 | +76.96% |
| 512 | 0.259148 | 0.137943 | +87.87% |
| 1024 | 0.558014 | 0.262755 | +112.37% |
| 2048 | 0.656312 | 0.528284 | +24.23% |
| 4096 | 1.102635 | 1.038863 | +6.14% |

These are best-of-20 exploratory tile selections. BF16_COMPARISON_32g.json separately aggregates the auxiliary BF16 linear timings across configurations: linear is much faster at M256/512/1024, but M4096 is 1.119945 ms versus current mm 1.102849 ms, a 1.53% throughput loss. Production route records for the new 4K profile are dominated by M4096; saved Python counts are not Graph replay counts or request counts. Therefore moderate-M local gains do not justify claiming a full-model win, and blanket INT8 decode would be a regression. Full-model Down INT8 would also need another 528,826,368 bytes of packed weights across 42 layers and fresh Math-500 validation.

Code, independent verifier, key JSON and CSV are in this repository. Full tensors and logs remain at /workspace/benchmark-results/20261021_int8_down_grouped_GPU_tianshu. No raw tensor payload is uploaded.
