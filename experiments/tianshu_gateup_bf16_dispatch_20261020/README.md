# GateUp BF16 dispatch component

Only prepares a separate component; does not change the frozen W8A8 runtime or overlap its GPU validation. Tests layer0 actual concatenated gate/up BF16 weights, M=32,40,48,56,128,512,565,576,1024,1537,1664,2048,4096. Includes unsupported decode batches, dominant prefill sizes, and rare odd/rounded sizes. Both methods compute the same BF16 linear operation: native flag_gems.mm(A,W.T) and actual selected flag_gems.linear(A,W,None). Operator source modules and SHA are archived; no monkeypatch/global registration.

Every method gets three phases with changed A and W through the same CUDA Graph; full GPU output is checked against TF32-disabled FP32 matmul, input guards and RNG are checked. Independent CPU verification recomputes archived 8x32 output selections per phase/method; it does not claim offline verification of full GPU outputs. Zero activation rows and M565/M1537 row strides are covered. Both APIs allocate their output so external output canaries are unavailable and explicitly recorded.

Nine alternating timing groups of twenty hot Graph replays per method and shape. Setup/autotuning excluded from replay timing, runtime compilation included in walltime only. This is not an end-to-end gain or a quality validation. The unpaired v3 screen that used a different prefill route was rejected; its score is only motivation for this controlled component, never a claimed comparison.

Run only after the exact current W8A8 validation pipeline is complete and GPU idle. Hard stop 2026-10-20 23:58 UTC+8. Raw tensor selections/logs remain in the Tianshu container; Git stores code, report, key JSON.
