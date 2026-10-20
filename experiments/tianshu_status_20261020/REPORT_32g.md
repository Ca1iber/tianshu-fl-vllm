# 天数BI-V150：2026-10-20进展（北京时间13:35）

当前采用的同一配置：**正式4K 5099.65、正式16K 3723.12 total tok/s，Math Level 3 102/105=97.142857%**。标签`tianshu-original-command-4k16k-k8v16-longseg-20261020`。原始serve argv、K软件8位/V BF16、12KiB物理页、4096预算、83预热点、严格TopP32K和M2048策略保持；新增长上下文B33..56的S16分段。完整[正式报告](../tianshu_long_segments_official_results_20261020/REPORT_32g.md)及[同配置入口](../tianshu_k8v16_longseg_release_20261020/README_32g.md)。

| 正式负载 | 官方基线总tok/s | 当前总tok/s | 输出tok/s | 平均TTFT ms | 相对基线吞吐 |
|---|---:|---:|---:|---:|---:|
| 4K/1K C64 N256 | 2028.01 | 5099.65 | 1019.93 | 4198.32 | +151.46% |
| 16K/1K C64 N128 | 915.15 | 3723.12 | 219.01 | 109729.32 | +306.83% |

两个负载各四轮排首轮，总1536/1536成功。4K后三轮极差/均值0.11373%，16K为0.06097%。相对上一已验证版本5101.45/3213.11，新4K变化-0.03528%、基本持平，16K提高15.87278%；这是历史完整评测比较。同源N64服务对照单独核验：4K+0.099%、16K+10.774%，所有384/384成功，见[4K](../tianshu_long_segments_results_20261020/REPORT_4K_32g.md)和[16K对照](../tianshu_long_segments_results_20261020/REPORT_16K_32g.md)。

[全新Math105](../tianshu_long_segments_quality_20261020/REPORT_32g.md)为102/105，105成功/0错误、全部自然stop。Math与原始双负载相同225源/46环境/原始argv通过核验，实际FL Graph捕获重放绑定通过。旧标签`tianshu-original-command-4k16k-k8v16-budget4096-20261020`保留为回退版本。新正式服务仍累计24次KV抢占；就绪后有一次0.51秒索引调优，未观察到该阶段GEMM调优日志，没有声称完全零现场调优。

长上下文分段组件按GPU实际长度≤8192保持4活跃段；较长上下文只在B33..56选择maxS16，其余维持原策略。B25..64完整扫描44场景、352 GPU参考和4224 CPU向量通过，B33..40组件约+53%、B41..48约+30%、B49..56约+13%。组件收益与整机分数分别记录，硬件波次解释仍是假设。

| 今天的其它探索 | 核验与结果 | 处理 |
|---|---|---|
| V定步长int8 / magic解码 | 全编码、数值及同Graph通过，B64/4K约慢32% / 46% | 不采用 |
| KV物理页padding | 96完整GPU参考通过，无整机收益证据 | 不采用 |
| M64 GEMM 49配置×4投影 | 196配置通过，QKV局部+12.6%，其它负收益 | 不采用 |
| 软件8位权重 | 32配置、288 GPU参考、73728 CPU标量通过，最快仍慢35%–52% | 不采用 |
| 同页标量页表读取 | 40 GPU/416 CPU通过，输出逐位同控制，B64/4K +0.013% | 不采用 |
| 动态K8+FP32尺度 / INT8 QK | 全codec、整数乘和44 GPU输出参考通过，11场景均慢23%–29% | 不采用 |
| GPU stream阶段计时 | 40帧，forward约88.2%，logits2.86%，sample8.56%；含host提交空隙 | 诊断 |
| 硬件计数profile | 两次工具失败，没有有效metric | 保留失败，不据此推断瓶颈 |
| 原生KV准入watermark 1/16 | CPU8边界及234源/48环境双模式通过，已在双负载结束后开始独占GPU服务对照 | 尚未采用 |
| K/V 32位打包读取 | 保持K8/V BF16格式与int64偏移，CPU全字节/位模式及源码差异检查通过；GPU组件准备完毕 | GPU尚未运行 |

KV水位实验见[service](../tianshu_headroom_service_20261020/README_32g.md)，打包读取见[packed-load](../tianshu_packed_loads_20261020/README_32g.md)。两者均不包含在当前发布配置，逐级数值/服务/新Math/正式结果通过后才考虑采用。

[沐曦最新提交只读检查](../tianshu_metax_reference_20261020/README_32g.md)未发现新的稳态可直接移植实现。沐曦仓库不修改。代码开发与评测均在天数容器；GitHub保存代码、报告、关键CSV/JSON，原件、张量、编译缓存留容器。当前最佳所需22个历史生成路径按3份不同内容归档，见[恢复入口](../tianshu_k8v16_restore_inputs_20261020/README_32g.md)；当前225源已核验，新克隆完整环境启动尚未验证。

14:25更新：[KV水位](../tianshu_headroom_results_20261020/REPORT_32g.md)六轮完整独立核验完成，抢占6→0，16K N64吞吐+0.02269%，不采用；[K/V合并读取](../tianshu_packed_loads_20261020/README_32g.md)12场景全部数值/同图检查通过，K32负收益、V32仅约+0.4%，未接服务。下一候选是缩小固定512MiB prefill暂存区，先验证容量边界及超容量原路径回退。

14:36更新：[小暂存区](../tianshu_small_arena_20261020/README_32g.md)CPU20边界、GPU22次完整staging/2次超容量分支检查已通过，释放384MiB+96KiB；1475715072活动元素逐位核验，fallback attention仍原样继承，未在本probe单独执行。236源/48环境服务对照正在串行4K→16K，未采用、无新Math。另准备[原生max_num_seqs缺省入口](../tianshu_seq_limit_default_20261020/README_32g.md)，11项真实AST CPU检查通过，旧64-seq微收益证据保留，今天尚未GPU复测。
