# 成对 K 读取：延后 V 加载、INT32 地址及其组合，四路对照

2026-10-21 14:21 完成。全部通过正确性检查，三个候选在目标 64 并发、4K/5K 组件上均更慢，全部拒绝，不启动服务、Math 或正式 benchmark。正式推荐仍为 `521c154` / `tianshu-original-command-4k-prefill-int8-20261021`，4K 总吞吐 5586.28 tok/s、输出 1117.25 tok/s、TTFT 3780.63 ms、Math 102/105；本报告没有新的模型成绩。

## 动机与范围

上一轮七组 T16 warp/stage 扫描虽然降低了寄存器数，但全部更慢。本轮检查另一种缩短活跃区间的方法：把原本在 QK/softmax 之前的 V 偏移、加载和类型转换延后到 P/alpha 之后、PV 之前。同时重测当前配对 K 读取内核的 INT64→INT32 地址及两者组合。

父读核字节与正式版本绑定：`qualified_current_reader.py` SHA256 `f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002`。旧地址实验基于更早的 `7b94...` 读核，4K 已是负收益；本轮仅检验当前配对 K 读核及与延后 V 的交互，不把旧实验视为收益。

四路为 `current / addr32 / late_v / late_v_addr32`。候选仅覆盖 25～64 个纯 decode 序列、3D、K uint8/V BF16、KV_NONE、T16/w1/s1、无 TD/窗口/特殊 mask 的明确范围。保留原 S 选择、原浮点表达式、原 host ABI。1～24 序列或任一不支持配置回退当前读核。

延后 V 的改动可逐字逆变换恢复父源码，关闭 constexpr 分支后的 AST 等于父源码。地址改动只涉及 28 个整数注解与 6 个 cast，AST 规范化证明无其他表达式变化，64 位指针和 writer slot_mapping 不变。带保守 masked 余量的 signed 元素偏移界限决定是否允许 32 位；producer 与 reducer 共享同一决策，任一不满足均共同回退原宽地址实现。

## 结果

每个值为 9 组交替顺序计时的中位数，每组 20 次同一 CUDA Graph replay。这里是单 Attention 组件耗时，不是服务 tok/s。

| 场景 | current ms | addr32 ms | late_v ms | late_v_addr32 ms |
| --- | ---: | ---: | ---: | ---: |
| B64 / 4096 | 0.552421904 | 0.566245747 | 0.595882463 | 0.604182673 |
| B64 / 5120 | 0.682384443 | 0.699883842 | 0.735861349 | 0.746560764 |
| B40 / 16384，仅组件 | 1.349830627 | 1.389585876 | 1.429343605 | 1.466859627 |

4K 耗时依次增加约 2.50%、7.87%、9.37%。5K 也均回退。4K 的寄存器分别为 176、167、155、146，所有方案均 0 spill。寄存器减少没有转化成速度收益；这组数据不单独证明 occupancy、prefetch 或访存延迟是原因。

## 验证与局限

- CPU 前置验证：2816 个实际 producer/reducer 选择与 token 覆盖案例、1760 个 capture/长上下文容量案例、84 个不支持/宽/负偏移共同回退案例、24576 个独立 paired-word→byte 地址样本；未初始化 CUDA。最初一次 CPU fixture 缺 NUM_SEGMENTS_PER_SEQ，补全 fixture 后 v2 全通过，发生在 GPU 启动之前，原日志保留。
- GPU：14 个案例 × 2 个原地更换 Q/K/V/table/lengths 阶段 × 4 路，共 112 次完整输出检查。离线重新加载并检查 8044544 个完整输出元素，所有候选输出与 current 逐位一致。
- 独立 CPU 重算 304 个 GQA 参考向量，核对记录的 1216 项 CPU 参考检查，满足声明的 `0.002 + 0.01 * abs(reference)` 容差；最大参考绝对差 0.002644539。它不能代替完整模型质量评测，本轮未运行模型。
- 504 组时序，覆盖 stride/ragged、短批次回退以及同一 Graph 从 capture 长度 1 切换至 4096/8193。记录实际 producer/reducer 源哈希、地址宽度、段容量、binary SHA、寄存器和 spill；缓存/scratch 守卫、输入未变、RNG 未变均通过。
- 超过 signed 32 位界限的回退只用 CPU stub 验证，没有分配超大 GPU cache，不能宣称实际大 GPU 分配已覆盖。
- `verify_gpu.py` 中保留了一条上一模板关于 tile reduction order 的旧注释；实际 verifier 在末尾强制所有完整输出 bitwise 相同，已通过。源已冻结，未为修正文案改动已执行文件。

源在 GPU 启动前冻结，结束重新校验全部哈希，runner PID 236962 终态 Z，GPU 已空闲。没有因观察超时重跑 GPU，也没有更改父目录或沐曦仓库。

## 复现与存档

代码准备提交 `1b2bc0f`，使用 `nohup env -u PYTHONPATH bash experiments/tianshu_pair_latev_addr32_20261021/run_component.sh`。脚本仅运行一次独立目录、1800 秒超时、预约 23:58 硬停，并在 GPU 完成后自动进行独立 CPU 复核。

关键结果：本目录 `CPU_GATE_32g.json`、`GPU_GATE_32g.json`、`GPU_ROUNDS_32g.csv`。原始结果 SHA256 `c0207052ad5f1cfaa1700c9bb21481b98220a34f80b0f8965a8d8480ef012f3c`。

完整张量、日志、源码快照只保留在容器：`/workspace/benchmark-results/20261021_pair_latev_addr32_GPU_tianshu`；runner 记录在 `/workspace/benchmark-results/20261021_pair_latev_addr32_runner_tianshu`。GitHub 只保存代码、报告、关键 gate 与小型 CSV，不上传权重、完整张量或大日志，不在本地留副本。
