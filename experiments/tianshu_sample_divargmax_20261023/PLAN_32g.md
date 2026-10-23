# FP32 采样除法与 argmax 融合组件

2026-10-23。实际 r4 GPU 时间线：16 帧 C64 纯随机采样，FP32 true_div rank2 约350us、argmax约87us。候选只融合原 inplace FP32 除法与局部 argmax，保留完整除法输出写回，第二步归约 tile maxima。固定 TILE4096/w4/s1，不改 softmax、TopP、RNG、模型、权重或服务。

先跑组件：B1/B8/B64、真实词表130560、非整除小词表、strided、普通/并列/NaN/无穷输入，两阶段动态输入、直接与图重放、完整除法位模式、原函数 token、输入 noise 不变、global RNG 不变。保存全部 CPU 输入/输出供独立 NumPy 审计，不以同一实现对照代替 CPU oracle。原后端 FP32 `/` 使用近似除法；CPU FP32 oracle 允许有限商误差2e-6，相对 token 选择则仍须与 CPU/native 完全一致；NaN 按首个位置、并列按首个最大值。完整结果也必须与实际 FlagGems 原函数逐位相同，不降低该阈值。

主性能点仅 C64 普通与strided。7轮交替，每轮30热Graph replay，均包含同样输入恢复，剔除首轮；完整 raw 时长保留。融合保留 inplace 副作用。正组件结果才考虑原服务对照；组件数字不是模型吞吐。截止北京时间23:58，硬alarm最多600秒。实验启动前冻结源，失败原件保留。
