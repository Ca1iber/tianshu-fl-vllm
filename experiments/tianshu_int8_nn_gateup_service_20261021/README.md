# GateUp NN 模型 4K 对照

继承原始 13 个 serve 参数、已通过质量的 359 份父运行源码、62 环境、4096 预算/64 序列/异步调度、K8V16/S16/FutureKV、LM8 BF16/LM64 NT INT8 和现有 BF16 prefill。只对 42 个 GateUp 的 M8/M64 使用组件选出的 NN 分块：M8 BM32/BN256/BK128/w4/s1，M64 BM64/BN256/BK128/w4/s1。模型精度、量化和逻辑权重值不变。

加载时在原生 DeviceMemoryProfiler 内将同一个逻辑 [N,K] bank 的物理布局改为列连续 [K,N] 后以转置 view 注册。逻辑形状仍 [N,K]，stride 从 [2048,1] 变 [1,12288]，GEMM 接收其 contiguous 转置。释放原 bank，无额外 INT8 权重 payload。每层完整设备转置一致、完整逻辑 SHA 与合格 43-bank 签名、BF16 权重版本、字节数和 profiler 上下文都实际验证；两组真实 KV 容量另行报告。

新路由记录包含实际 NN/NT 内核源码、选中配置、二进制、寄存器、spill、42 层 M8/M64 真捕获调用。旧 dense binding 的 kernel 字段只能代表原 NT 模块；新增 NN_gateup_record 才是 GateUp 实际执行证据，不把 Python 调用数当 Graph 重放数。当前 LM head 全部保留现有路径。

每组 4K/1K、C64/N64 三轮排除首轮，成功数/usage/seed canary/原生边界/完整冻结源码/FutureKV/抢占统一检查。GPU 组件不是模型质量；若稳态收益明确，再对最终配置跑全新 Math 与正式 4K。无常规 16K。原始日志只容器，代码报告关键结果推 GitHub，23:58 北京时间硬停止。
