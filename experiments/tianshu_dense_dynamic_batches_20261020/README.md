# 动态批次真 INT8 GEMM 组件筛选

当前正式 W8A8 只命中 M8 和 M64；其他动态批次仍走 BF16。新组件对原始 layer0 GateUp 和 LM head 的 M32/37/40/48/56/63，分别筛选四个已验证可编译配置。M37 对应 16K 内存约束下可能出现的真实批次，M40 对应补齐图尺寸；此处不声称服务已经出现这些路径，也不例行跑完整 16K。

每次 GPU 计时都包括完整 A 动态量化和 INT8 GEMM；静态 W 打包在计时外明确单独记录。GateUp baseline 为现有 flag_gems.mm，LM head baseline 使用实际 vendor flag_gems.linear。每配置三次改变输入/权重/scale，全矩阵整数结果与量化输出在 GPU 核验，并检查输入、RNG 和边界；7 轮交替顺序，每轮 20 次热 CUDA Graph 重放。离线 CPU 复核所有 A 量化与选定原权重行/输出坐标，不能称为离线全矩阵输出复查。

复用字节一致的 b9b692c 已合格整数原语，CPU gate 只证明通用安全界及来源一致，新 M 的 GPU 正确性需要本实验完成后才成立。全部原始 bank SHA 与合格 Math 比较。不是模型评测、不是正式吞吐，也不自动采用最佳微测配置。排在预填充服务及其质量/正式验证之后，等待期间不占 GPU。

LM 基线解释：已追到现有 UnquantizedEmbeddingMethod.apply 调用 dispatch_unquantized_gemm/default_unquantized_gemm，后者调用 torch.nn.functional.linear。F.linear 在 FL 分发下的最终算子还需真实服务记录确认。本组件显式调用 flag_gems.linear 作为 BF16 比较对象，不能直接把组件差值当作现有服务 LM fallback 的收益；最终仍需服务 A/B。
