# W8A16 M64 Down initial GPU screen — negative performance

All sixteen frozen tiles/layouts completed two new layer-0 M64 fixtures. Independent CPU full-matrix FP64 reconstruction checked 4,194,304 output cells against the predeclared FP32 accumulation plus BF16 rounding bound; all passed. For representative NN64x64/BK128/w4, relative L2 versus ideal original BF16 dot was 1.01384% and 1.01900%; relative L2 versus ideal W8A16 dot was 0.16561% and 0.16576%, dominated by BF16 output rounding. This component gate does not establish Math accuracy.

Primary control is the actual released-service Iluvatar BF16 launcher, not torch.mm. Whole-MLP graph medians: service BF16 0.117309618 ms, secondary torch.mm 0.138727689 ms, best W8A16 NT64x64/BK128/w4 0.169617701 ms. Down-only medians: 0.0592038453, 0.0751275003, and 0.0999888480 ms respectively. Current W8A16 is substantially slower; no 42-bank rollout, Math evaluation, official benchmark, or adoption was done.

Source/checkpoint/input/weight/RNG/canary and poisoned graph/eager identity guards passed. Read INDEPENDENT_CPU_GATE_32g.json and KEY_TIMINGS_32g.csv for all variants and numerical/error-bound details. Raw arrays and cache remain in the container. Future versions may diagnose conversion and tiling costs, but must keep this failed performance result intact and use new source directories. Published r2 is unchanged.
