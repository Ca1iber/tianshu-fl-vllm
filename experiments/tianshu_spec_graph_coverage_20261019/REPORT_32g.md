# 天数投机请求图覆盖率（2026-10-19）

实际原生 ngram1/fixed-index/正 FP32 标准 rejection/Q2split 路径的两个 4096/512、C64/N64 诊断阶段完成。128 负载请求与 4 canary、91 源和42环境、6零抢占指标全量下载独立审计通过。仍非正式4K/1K，新Math未跑，未采用新版本。观察器包含 marker 文件读取和定期JSON写入，未测开销，不将其吞吐当优化分数。

第一轮77.535269秒、client TTFT18424.429ms，4次M311 GEMM调优合计日志1.53秒；第二轮72.366978秒、TTFT14841.961ms，零新增调优，各64/64成功。诊断total tok/s为3803.585/4075.229，仅用于记录本次4096/512负载。

|真实execute类别|第1轮|第2轮|实际图分支|
|---|---:|---:|---|
|已完成prompt的混合Q1/Q2|472|464|全部NONE/eager|
|已完成prompt的全Q1|26|47|全部NONE/eager|
|prefill或尚有prompt未完成|131|131|全部NONE/eager|
|空execute|1|1|无forward|

合计1009次decode/verify，936次混合Q1/Q2、73次全Q1；真实请求没有一次FULL replay。启动/显存测算等execute之外另有50次FULL dispatch/capture与52次NONE pass-through，这仅是成功FL GraphWrapper API分支调用次数，不是50个不同运行时图，也不是GPU逐kernel trace。计数严格排除seeded/flush canary和未sealed阶段。

实际运行async=False、specasyncFalse（model记录），75点prefill补齐与hybrid被spec guard关闭；不能把所有退步归因某个attention kernel。两轮draft/accepted计数为5996/5405与4314/3808；接受率高不等于每步都在投机。Source _is_uniform_decode要求Q2且tokens==2*requests，当前全部decode均不满足运行时覆盖。

下一候选按真实token/request分布，为N64的混合Q1/Q2建立uniform=False独立图，容量72/80/88/96/104/112/120/128；对应eligible与padding明细保存新folder COVERAGE_PLAN_32g.json。先CPU原生dispatcher/元数据边界，再真实模型graph-vs-eager hidden/logits与KV写入检查，不能直接force uniform复用Q2图。当前尚无候选GPU结果或收益。

完整原件 local service-results/20261019_tianshu/spec_graph_coverage/20261019_spec_graph_coverage_tianshu，remote同名benchmark-results。20parts/5,102,650B归档SHA256 ed122743eb69fbc255d23b8dda8173c6257be453e264bbb7577e357c0b3166b2，分片与整体已核验，source HEAD d3c7d96。所有GPU进程已退出。原best4093.81与历史独立Math102/105保持；无新16K/push/agents/MetaX改动。
