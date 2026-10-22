# V1 signed-zero reference failure

原 V1 GPU 测试在第一个尾部用例的 phase1 停止，未产生计时/收益结论。原始源码和目录保留。单独 GPU 诊断对 3 种格式、3 个相位重新解包，所有不一致均为浮点 -0 与整数零解码所得 +0；没有非零值不一致。

V2 不修改量化或 kernel：将“负号乘原解码浮点值”的期望零符号修正为实际整数码解包规则的 +0。独立 CPU 解码哈希采用相同规则，仍逐位比较全部元素，不放宽非零误差。诊断 9 个场景结果见 SIGNED_ZERO_DIAGNOSIS_32g.json。

V1 原始目录 `/workspace/benchmark-results/20261022_quant_weight_component_GPU_tianshu`；V2 使用新目录，V1 原 failed 状态不覆盖。
