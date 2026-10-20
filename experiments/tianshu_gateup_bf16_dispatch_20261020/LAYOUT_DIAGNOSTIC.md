# GateUp BF16 layout diagnostic after numerical failure

The first13-size probe stopped at M565 with row-strided A; six preceding sizes completed. The failure is preserved at the original raw root and is not a service result. Do not relax tolerance or assume which method failed. This separate diagnostic tests M565 and M1537 in both contiguous and row-strided layouts, plus the actual dominant contiguous M2048/M4096 prefill sizes. It leaves all five originally frozen probe files unchanged.

Keep atol.02+rtol.01. For every method and three changed-input phases, record full GPU failure count. Archive actual violating pairs and independently recompute those FP32 CPU dots. Failures remain failures, and numerically failing cases are not timed. Timing runs only if both methods pass all phases, nine alternating20-replay Graph groups. No runtime installation/adoption. Only the explicitly tested input/layout contract can support a future candidate; the failed layout is never silently discarded.
