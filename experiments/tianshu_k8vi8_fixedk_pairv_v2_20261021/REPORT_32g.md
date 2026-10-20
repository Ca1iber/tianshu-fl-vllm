# KV V8: fixed K decode and paired V loads, BI-V150 32G, 2026-10-21

Not adopted. All ten measured scenarios are slower than the current K8/V16 implementation. Model/Math/full benchmark were not run for these negative candidates.

V1 (tianshu_k8vi8_fixedk_pairv_20261021) failed before completing a scenario: a tl.static_assert tested three runtime stride values and the compiler could not determine its condition. V1 source and raw failure logs are retained. The independent V2 removes that one assertion and enforces positive even strides, D128, contiguous final dimension and KV mode 2 in the actual host launch before any paired uint16 load. CPU checks exhaust all 65,536 byte-pair decodings, dimension mappings and valid/invalid host guards.

Four paths use the actual writer, page layout and block table: current K8/V16; existing per-head V8; fixed soft8 K decode omitting K-scale reads; and fixed K plus paired V loads. The K-scale storage/writer is unchanged. Ten scenarios, three varied phases and four paths produced 120 successful GPU checks. The CPU verifier recomputed 1,248 selected reference vectors and independently requantized 907,776 actual writer V values. Guard/input/RNG checks passed. Seven rotating rounds of 20 graph replays include the writer, attention reader and reduction.

| C64 context | K8/V16 ms | existing V8 ms | fixed K V8 ms | paired V8 ms |
|---:|---:|---:|---:|---:|
| 4096 | 0.540140 | 0.752010 | 0.747898 | 0.752596 |
| 5120 | 0.668391 | 0.936099 | 0.931325 | 0.938100 |

At C64/4096, fixed K increases elapsed time by 38.46%; paired V increases it by 39.33%. Registers are 176/189/187/186 respectively, with zero spills; these metadata do not establish a causal hardware explanation. V8 reduces per-page storage from 12,288 to 8,448 bytes but does not improve measured runtime. Keep the validated K8/V16 production path.

Key results and full source snapshots/hashes are in GPU_GATE_32g.json and ROUNDS_32g.json. Raw fixtures and failure logs stay in the container under /workspace/benchmark-results/20261021_k8vi8_fixedk_pairv*_tianshu.
