# QKV/O actual unquantized BF16 route

A fresh process mirrored WorkerFL's FlagGems enable-all with empty iluvatar blacklist. The installed UnquantizedLinearMethod.apply resolves to F.linear and the warm Python call trace enters the original vendor linear.py (line117), not explicit mm. Both returned outputs were retained directly in CUDA Graph, with no artificial output copy. All8 cases/two independent fixtures/7 alternating rounds pass source/checkpoint/input/weight/RNG/poisoned-graph checks. Full BF16 outputs equal explicit vendor mm for all tested inputs. Independent CPU recomputed8192 selected FP64 output comparisons with a predeclared FP32/BF16 bound.

Actual-apply graph ms for QKV M64/128/512/4096:0.04162942/0.03526077/0.10148789/0.50072346. O:0.03229731/0.02464519/0.05893692/0.39322386. The explicit mm control differs in timing even when outputs match. Thus the promising O-M128 native timing (~0.028ms) does not establish benefit over actual F.linear (~0.0246ms); exclude that point from a future model version.

An actual all42-bank F.linear control is the next confirmation. This diagnostic does not claim to reproduce a full compiled model/Graph path; a future service must audit actual fallback and runtime calls and compare matched short4K with the published r2. Original raw diagnostic/source unchanged. No model/quality/official benchmark/adoption.
