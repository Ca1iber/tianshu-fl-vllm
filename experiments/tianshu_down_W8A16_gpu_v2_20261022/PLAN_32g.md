# W8A16 M64 v2 — match actual-service warp parallelism and pipelining

V1 retained intact: full independent numerical gate passed, but its best Down0.09999ms versus actual vendor0.05920ms is negative. Read-only persistent BF16 tuning cache for (64,2048,6144) records BM64/BN64/BK128, 16 warps, 2 stages, SPLIT_K1. V1 tested 2/4warps and stage1, so it did not test that parallelism.

New immutable version, same W8A16 math and full CPU error gate, new independent seeds202610231200/201. Fixed16 variants: NT/NN,64x64, BK64/128,8/16warps,1/2stages. Kernel new unique JIT name. Actual-service BF16 primary control, torch.mm secondary, unchanged GateUp64 and original checkpoint. Before GPU freeze all897 inputs including890 exact r2 model sources. Numerical gamma6145FP32 plus output-half-BF16-ULP and FP64 bound identical to V1; all full outputs checked independently. Source/weight/RNG/canary/checkpoint and poisoned graph/eager guards unchanged.

Seven alternating rounds x20 replays per variant, both Down-only and whole-MLP, eager/graph. Resource or correctness failure preserves all raw results; no post-GPU source patch. Single layer0 cached screening only; 42-bank holdout/fresh full C8/C64 Math>95% required for future promotion. No benchmark/model executed here and no adoption. Max30min, hard stop23:58BJT. Code/reports/key results GitHub, raw arrays/cache container only.
