# 原 single-dot 融合失败，2026-10-21

原始实验总体失败：M8 contiguous 的4个双 dot 配置完成，首个 single-dot 配置在完整实际计时输出的严格逐位比较失败。保留失败原始状态，未放宽阈值、未将部分成功冒充整轮通过。

四个双 dot 配置的已保存输出由独立 CPU 恢复项目 `../tianshu_nn_silu_partial_cpu_20261021` 复核；其证明范围仅限已完成数据，无 GPU 重跑。后续完整双 dot M8/M64/stride 试验另见 `../tianshu_nn_silu_two_dot_20261021`。

冻结 single-dot 实现由 `../tianshu_nn_silu_single_debug_20261021` 单独保存原失败实际二进制、完整阶段输出并作 CPU INT64 诊断：原生整数结果与CPU一致，而单dot的Up半部有43,008个整数不一致，属于数值计算错误。该方案不采用。

原始目录 `/workspace/benchmark-results/20261021_int8_nn_gateup_silu_GPU_tianshu`。原始 result.json SHA256 为51e485645fbb17deb3811d8e04f32180a527f435ca0f0067e00b482562c10969。
