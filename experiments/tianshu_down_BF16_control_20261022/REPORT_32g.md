# M64 actual-service BF16 control diagnostic

The r2 Down policy calls the unchanged Iluvatar FlagGems module mm(x, weight.T). Its existing M2048 wrapper delegates M64 to the original _launch_mm. Frozen official sources and their SHA256 were checked before and after execution.

On one layer-0 M64 random BF16 activation fixture, seven alternating CUDA Graph rounds (100 replays each) measured torch.mm median 0.0743498421 ms and the actual service vendor launcher 0.0595935011 ms. The launcher writes preallocated output, avoiding an artificial copy; graph device timing excludes CPU allocation. This is a baseline diagnostic, not a model or official benchmark.

Both paths were finite and graph/eager exact individually. They differ at 65/131072 BF16 output cells (relative L2 0.0000592290, maximum absolute difference 0.015625), so this diagnostic does not claim the two BF16 implementations are bitwise interchangeable. Original checkpoint, input, weights, RNG, and frozen source guards passed.

Previous component torch.mm controls remain valid descriptions of those tests, but their small speedups cannot be interpreted as speedups over the actual released BF16 service. R2 formal end-to-end comparisons remain unchanged and valid. No candidate was adopted here.
