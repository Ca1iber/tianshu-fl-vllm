# RMS -> A8 fusion component: too little main-batch benefit for model integration

Not adopted. Actual checkpoint layer0 RMS weight, M8/64/4096, three ROWS/warp configurations, three changed-input phases, same materialized BF16 output plus row A8/scales. No model/FX hook or model accuracy test was installed. All27GPU phase/config checks and independent CPU fixture checks pass. CPU quantization uses every saved phase2 normalized BF16 element; the FP64 RMS mathematical reference covers selected rows only. Comparison to current compiled RMS uses a declared tolerance, not a bitwise-equivalence claim.

| M | compiled RMS + original A8 ms | best fused ms | component speed gain |
|---:|---:|---:|---:|
|8|0.012011923|0.011631346|+3.27%|
|64|0.012010577|0.011925962|+0.71%|
|4096|0.143924999|0.135577500|+6.16%|

Seven rotating rounds of20Graph replays include the actual compiled normalization and qualified div_rn/RNE quantizer in the baseline. Candidates materialize normalized BF16 too, preserving potential fallback data. ROWS4 variants regress at all three sizes. At dominant decode M64 the best absolute saving is only0.0000846ms per layer; at M4096 it is0.00835ms per layer. These are component savings, not measured model throughput, and do not justify a complex FX integration ahead of the independently promising budget8192 candidate.

Native IR text includes a BF16 cast before RMS weight multiplication; actual generated model code elides it. Both actual standalone compiled traces and mathematical/IR deviation counts are retained in the raw archive. Future fusion must identify the executed arithmetic, expose quantization and residual/fallback data explicitly in FX and verify real capture/replay. No global tensor side channel is used or proposed as an implementation.

Raw tensors and compiled traces remain at /workspace/benchmark-results/20261021_rms_a8_GPU_tianshu. Code, key gate, rotating timing JSON and CSV are retained here. Fresh Math>95% would still be required if any model integration were attempted.
