# Down K-group M64 layout/tile screen v4

16 predeclared variants (G128/G512, NT/NN, 32x128/64x64, 2/4 warps) completed two independent layer-0 fixtures. The frozen GPU checks and independent CPU INT64/libm.fmaf reconstruction passed. See INDEPENDENT_CPU_GATE_32g.json and KEY_TIMINGS_32g.csv for counts and full rows.

Whole-MLP graph median: torch.mm BF16 control 0.138383853 ms; ordinary native A8W8 0.128708458 ms; best grouped G512_NN_m64n64_w4 0.136988854 ms (+1.01833% relative to that functional BF16 control). No 42-bank, model-quality, or model-throughput claim follows from this single-bank cached screen.

A subsequent separate actual-service BF16 diagnostic (tianshu_down_BF16_control_20261022) found the published service calls the Iluvatar FlagGems launcher at M64. That launcher is faster than torch.mm in its own alternating test. Thus the 1% above does not establish improvement over the released service; do not promote this screen. Cross-run times are diagnostic only. Original results and frozen sources remain intact. The published r2 tag is unchanged.
