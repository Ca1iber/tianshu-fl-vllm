# QKV/O weight-group precision — CPU-only lead

Original84 real banks, two new synthetic BF1632row fixtures,64 stratified output columns per bank,504 bank/phase/variant records. Separate Q/K/V regions are represented; this is selected-column synthetic analysis, not an estimate of model-level accuracy or an exact full-vocabulary error. Baseline full per-row activation quantization and old layer0 native selected INT64/BF16 evidence independently matched624 cells. Original checkpoint and frozen script/plan hashes unchanged.

Weighted relative L2 across the selected outputs:

|Role|Existing per-output W8|K512 ideal grouping|K128 ideal grouping|
|---|---:|---:|---:|
|QKV|1.331153%|1.211431%|1.104185%|
|O|1.234171%|1.141555%|1.052473%|

K512 reduces this synthetic error by8.99%/7.50% relative; K128 by17.05%/14.72%. Grouped partials use exactINT64 and idealFP64 weighted accumulation, not a claim to reproduce future GPU FP32 FMA. Static grouped codes were computed only for the selected output rows; no complete new GPU bank is qualified.

For all84banks, total weight-code bytes would remain396361728. Existing scales774144bytes, K512 scales3096576 (+2.214844MiB), K128 scales12386304 (+11.074219MiB). These are shape-derived storage estimates, not measured GPU allocations. A fused group-scale GEMM with independent arithmetic and timing still needs design; monolithic native INTMM cannot apply separate group scales. No model/dataset/GPU execution/adoption here. The active903source QKV/O Math candidate and published r2 remain unchanged.
