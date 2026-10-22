# W8A16 v2 higher parallelism — still negative

All16 high-parallelism variants completed independent numerical checks (4,194,304 full output cells) with the unchanged mathematical FP32/BF16 bound. Primary service BF16 whole-MLP graph median0.117659998ms, best candidate NT64x64/BK128/16warps/stage1 0.147607493ms. Down-only0.0589875042ms versus0.0784555733ms. Even matching the vendor BF16 warp count does not make this fused weight conversion implementation competitive. No model quality/throughput test or adoption.

Single-bank cached screen; full42-bank memory-traffic behavior has not been established. Sources, checkpoint, poisoned graph/eager, canaries, input/weights/RNG checks passed. Full table/error details in INDEPENDENT_CPU_GATE_32g.json and KEY_TIMINGS_32g.csv. V1 and qualified r2 remain intact.
