# RMS+A8 GateUp与LM pad32..64组合：首次完整双质量通过

| 新鲜Math-500 Level3 | 正确/总题数 | 正确率 | stop/max_tokens | 错误索引 |
|---|---:|---:|---:|---|
| C8 | 102/105 | 97.142857% | 105/0 | 18、26、87 |
| C64 | 101/105 | 96.190476% | 105/0 | 18、23、87、98 |

两批均严格>95%，每批105请求全部成功、原始evalscope退出码0，无cap、无错误或未完成请求。每个版本只跑首次新鲜质量，没有复用父版本分数，也没有同版重跑挑数。C64沿用既有FutureKV control质量政策，C8为原candidate政策，未为了此组合修改政策。

944项模型输入在GPU前5987cf1冻结，含原RMS916与LM925完整父集合的并集及组合接入；共同原r3的903项精确字节保留，91项质量控制器也在首次GPU前冻结。数值变更仅静态M64 GateUp前42层IR-native RMS+A8融合与LM pad32..64及显式BF16舍入后FP32输出。原checkpoint和结构、13serve参数和63环境契约、QKV64/O64/Down64及LM<=8等原BF16路径保留，未加入QKV64量化。

独立CPU读取每批全部105条prediction/review/report重新计分，核对全部模型/控制器及首次GPU前快照、原checkpoint SHA；重新执行所有继承attention/KV/scheduler/sampling/GU/LM/Down/QKV/O与新RMS/LM实际运行审计，完整profile和accuracy_audit均与原始严格相同。真实42层RMS/原GU图捕获与CUBIN、LM新二进制/原pack/Profiler峰值和C64实际M64运行均通过；旧独立A8与旧LM64执行未冒称。RMS组件记录的“LM未改”只指RMS钩子本身，组合整体LM已明确替换并独立审计。

C64实际QKV/O与Down M4096均42层，QKV同时命中M128。C8本轮真实请求未命中QKV/O native INT8运行分支，不能冒称C8完成M128/M4096运行覆盖；组件启动/图捕获记录与真实请求运行记录分开检查，实际prefill覆盖由C64提供。

QUALITY_GATE.complete和model_quality_qualified均true。同期r3控制与组合候选的N64×3排首轮短4K已经串行启动，目前控制组预热，尚无组合吞吐结果。仅短测增益>0.5%、候选spread<1%、TTFT<=对照1.01且独立运行审计通过，才条件执行已冻结完整原N256×4的4K；不运行16K、不自动发布。当前正式最佳仍为r3的5950.75 total tok/s；单独RMS +0.1344%与单独LM +0.4378%不能直接相加当作组合收益。

完整原始响应、日志、数组、源快照及缓存仅天数容器/workspace/benchmark-results/20261023_RMS_LM_math{8,64}_tianshu。GitHub只保存代码、逐题关键分数、独立门槛和报告，本地没有模型/评测副本。
