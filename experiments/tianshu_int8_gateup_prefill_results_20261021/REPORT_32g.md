# Prefill GateUp 真 INT8 NN：六种 M 组件通过

2026-10-21，BI-V150 32G。当前模型 M≥128 的 GateUp 仍是 BF16 linear_contiguous；此次组件以同一个实际 vendor BF16 linear(a.contiguous(),w,None) 为共同基线，没有用早期较慢 mm 来夸大提升。真实 layer0 GateUp，W 静态 NN 布局，六个预声明分块；完整 A 动态量化计入每次热 Graph，静态 W 打包排除。

36 个组合 / 108 次变 A、W 符号和 scale 的完整 GPU INT32 与量化 BF16 检查通过。GPU FP32 整数参考先要求实际 sum(abs(products))<2^24；另有全部 A 量化、真实 checkpoint 选中 W 重新量化与实际 GPU 选中输出 CPU INT64、guard、输入/RNG/变数据 Graph 重放检查。独立 CPU 复核通过，原始矩阵完整 GPU 验证和 CPU 选中输出验证的范围明确区分。

| M | BF16 linear ms | 最佳 NN INT8 ms | 基线/候选-1 |
|---|---:|---:|---:|
| 128 | 0.131278 | 0.095573 | +37.36% |
| 256 | 0.215222 | 0.159256 | +35.14% |
| 512 | 0.316759 | 0.269348 | +17.60% |
| 1024 | 0.582194 | 0.531248 | +9.59% |
| 2048 | 1.172077 | 1.052624 | +11.35% |
| 4096 | 2.362187 | 2.089838 | +13.03% |

每组七轮交替次序、每轮20次 Graph。表格是六个候选里的探索性最佳，完整分块和资源元数据见 GPU_COMPARISON_32g.json；未作独立 holdout 或模型吞吐承诺。量化对 BF16 的误差独立记录在 APPROXIMATION_32g.json，不把 INT32 计算正确当成量化模型准确率达标。当前没有模型接入、没有新 Math、没有新正式 benchmark。

继续测试 grouped-M program 顺序与 group1 的成对组件，以评估大 M 的权重复用机会；随后才决定模型接入。下一版本若启用 prefill A8/W8，必须重新跑105题 Math，严格>95%，不能沿用旧 decode-only 量化的成绩。

完整 raw /workspace/benchmark-results/20261021_int8_gateup_prefill_GPU_tianshu，仅容器；代码、报告、关键 JSON/CSV 上传天数 GitHub。新操作未改变当前推荐正式4K 5482.79或吞吐备选5509.23。
