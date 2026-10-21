# 分段 load + join 单 dot 融合，2026-10-21

结果：16 配置、48 个变化输入 Graph 阶段全部通过严格逐位比较和独立 CPU 校验，编译失败0；性能显著落后，放弃此实现。

M64 连续输入最佳实测：控制0.0854967 ms，join融合0.2783531 ms；M8 控制0.0941517 ms，join融合0.2791896 ms。未接入模型，未运行服务 benchmark 或新鲜准确率。

此次把 Gate/Up 分别连续加载后用 `tl.join(...).reshape(BK,2*BN)` 交错，再拆分累加器，避免原单 load 版本的整数列错误。正确性修复不代表性能改善。保留双 dot 正确且更快的实现作为模型接入候选。

原始完整输出、输入和计时保留容器 `/workspace/benchmark-results/20261021_nn_silu_join_GPU_tianshu`；独立校验范围、源与二进制 hash 见 `GPU_GATE_32g.json`。
