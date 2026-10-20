# K8 + dynamic V8 with integer P×V: independent component prototype

The earlier dynamicV8 writer/layout passed full physical/CPU checks, but its BF16 consumer was about39% slower than K8V16. This prototype changes the PV computation: fold per-token V scale into P, quantize each query/tile row of weightedP with IEEE FP32 RNE, use INT8×INT8→INT32 dot, then rescale into the existing FP32 online-softmax accumulator. QK, softmax/reducer, paired software8 K and the real V8 writer/layout are preserved. The flag defaultsFalse so the oldV8 path remains available as a control. Only3D single-token decode without sliding window is intended; tileT>=32.

This deliberately changes attention approximation. CPU/source validation cannot replace GPU checks or fresh Math>95%. Plan: first compile a primitive diagnostic and compare full quantizedP/scales/INT32/output to independent CPU oracles, then full real-writer attention and changed-input graph cases, with CPU FP64 attention checks and nativeK8V16/nativeV8 baselines. Only clear component gains justify engine integration; no GPU task is scheduled yet, and the running budget test/conditional Math pipeline is not interrupted.

No installed or qualified source is modified. Code/raw only on Tianshu container; GitHub stores code/report/key data. CPU_GATE explicitly does not claim GPU correctness/performance or model quality.
