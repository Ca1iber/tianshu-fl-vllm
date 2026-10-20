# 真 INT8 权重 NT→NN：GPU 组件完成

2026-10-21，BI-V150 32G。只把权重从 contiguous [N,K] 静态转置成 [K,N] 并改变载入寻址；每次 A 动态量化、整数点积、mask、scale、输出、selected tile 保持相同。W 转置/打包/上传单独记录、不包含热 Graph 计时；A 量化与 GEMM 都计时。

28 场景×3 次改变输入/权重符号/scale，84 组完整 GPU INT32、BF16 逐位一致；独立 CPU 重读 checkpoint 选中原行量化、全部 A 量化、选中输出 INT64、编译源和快照核验通过。每种场景七轮交替次序，每轮 20 次 Graph 重放。包含 M1/3/7/8/17/32/37/40/48/56/63/64，strided37 与 N130/K259/strided17 双尾部。GPU 参考用 FP32 点积，但仅在实际绝对乘积和 <2^24 时作整数精确性对照；没有用 GPU FP64。

| 场景 | NT ms | NN ms | NT/NN-1 |
|---|---:|---:|---:|
| GateUp M8 | 0.092502 | 0.091384 | +1.22% |
| GateUp M64 | 0.067890 | 0.064652 | +5.01% |
| LM head M8 | 0.876706 | 0.855666 | +2.46% |
| LM head M64 | 0.629520 | 0.666613 | -5.56% |
| GateUp M32 | 0.060228 | 0.065039 | -7.40% |
| LM head M32 | 0.556522 | 0.601711 | -7.51% |

结果依赖 role/shape，不能统一采用 NN。当前模型只在 GateUp M8/M64 与 LM64 使用真 INT8，LM8 是 BF16；LM8 的整数组件结果不能据此改变模型精度。下一步优先 GateUp 的 NN 分块选择，若实际收益值得再做 4K 服务对照。本轮不执行模型、Math、正式 benchmark，不产生模型新成绩。

全部 28 项中位数在 KEY_TIMINGS_32g.csv，逐轮在 ROUNDS_32g.json，完整依赖/二进制元数据与数值门在 GPU_GATE_32g.json。原始 PT、硬件、源码快照与 result.json 仅容器 /workspace/benchmark-results/20261021_int8_weight_nn_GPU_tianshu，GitHub 只保留代码、报告和关键 JSON/CSV。
