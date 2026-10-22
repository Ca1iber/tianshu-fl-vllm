# Grouped A8W8 v5 actual-service control — no whole-MLP win

All16 variants and two independent M64 fixtures passed full native group INT32/separate-FMA GPU equivalence and independent CPU INT64/libm.fmaf projections. Control is the actual service Iluvatar BF16 launcher.

Whole-MLP graph median control0.118426931ms, ordinary nativeINT8 0.127652311ms, best grouped G128_NT_m64n64_w16_s2 0.129692304ms. Down-only some grouped configurations approach the control (best G512_NN_m64n64_w8_s1 0.0591899991ms versus control0.0597726882ms), but this does not survive inclusion of unchanged GateUp64. Thus isolated GEMM/Down timings do not establish service improvement.

Single-bank cache screen only. No model/Math/official benchmark or adoption. All source/checkpoint/graph/eager/RNG/canary guards passed; full results in INDEPENDENT_CPU_GATE_32g.json/KEY_TIMINGS_32g.csv. Full42-bank traffic and fresh quality would still be required. Original v4 and published r2 are unchanged.
