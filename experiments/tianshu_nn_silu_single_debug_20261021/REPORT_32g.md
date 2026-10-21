# 冻结 single-dot 错误的独立诊断

诊断完成不代表算法通过。`DEBUG_GATE_32g.json` 标记原算法失败、未采用。实际失败计时二进制与原实验一致；保存全部阶段输出后独立 CPU INT64 复核98,304个值。原生INT32及BF16投影无误，单dot Gate半部无误，Up半部43,008个INT32值错误，故不属于可接受舍入差异。实际激活输出42,927个BF16元素不同，最大绝对误差119.6875。

完整张量 `stages.pt` 只保留容器 `/workspace/benchmark-results/20261021_nn_silu_single_debug_GPU_tianshu`，SHA256 2a37db46b5ffbc26eb83dd1adcbc5825a95fa5cf41d363b170a872f221f64e7f；GitHub保留源和关键诊断gate。具体编译器内部根因尚未由IR证明，不归因到特定编译优化pass。
