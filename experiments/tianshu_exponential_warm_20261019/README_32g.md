# 天数32G：采样指数噪声预热覆盖

本项研究冷启动稳定性，未计新吞吐或新质量。已验证正式4K首轮仍出现 N8094720=62×130560 的指数噪声现场autotune，4候选+1总结（打印0.66秒）；75点 GEMM 预热并未覆盖这个算子。后三轮没有新增调优，不将0.66秒解释为整轮冷启动差距。

warm.py拟以单个FP32 64×130560缓冲，覆盖B64至B1实际.contiguous().exponential_形状，在真实Torch fork_rng中保存/恢复CPU与默认CUDA RNG，禁止Graph捕获。每请求独立Generator不传入helper，不能仅凭CPU假对象宣称实际RNG完全一致。

待GPU组件：使用全新隔离的FLAGGEMS_CACHE_DIR/DB，仅让调优配置冷、保留现有编译缓存；实际FlagGems路由记录N和编译BLOCK，保存warm前后日志，证明后续B1..64没有新增指数调优。随后逐token对照原native random_sample，在全局、混合种子、全种子情况下比较token、默认CPU/CUDA及每请求Generator最终状态，并在warm成功/失败路径检查恢复。此验证不声称冷配置重新选出的BLOCK永远相同。

只在Math/正式GPU任务退出后做组件，不并发、不清空共享缓存、不改安装源、不自动接入当前最佳。未测GPU、服务、Math、硬件profile，正式best4997.05不变。
