# 新LM1..64 W4 vs精确r3同期短4K

质量尚在运行时提前冻结筛选协议与控制器，不抢跑GPU。GPU前必须有相同944-source模型、fresh完整C8/C64各105题均>95%、独立逐题/原始/源快照审核通过的QUALITY_GATE；控制r3精确903-source，候选944，13argv/完整63环境仅PYTHONPATH不同。不借用旧934候选质量，不使用CPU恢复数据当正常新run。

4096输入/1024输出/C64/N64，control再candidate，各3轮，首轮预先排除，全部请求成功且长度/种子/usage逐个相同，0preempt/0运行时autotune。冻结screen：热轮mean gain>.5%、候选极差/均值<1%、TTFT<=同期control*1.01。每个完整轮次向用户汇报；全部计时轮次CPU空闲，不写代码/推送/打包/校验/去重，只轻量poll。双方结束后才独立CPU重算源/控制器快照SHA、全运行审计和计时/TTFT，所有负结果保留，不原样重跑。

C8旧BF16 LM要求及M64 W8要求已经显式替换为实际W4调用，不改旧counter，GU/attention/KV/scheduler/sampling/Down其他范围及真实峰值检查保留；原control审计独立实例不受候选适配。候选Down42真实W4捕获联系实际B64模型replay，LM64至少100次真实主图外调用；LM8烟测也需实际W4。组件阳性不当模型成绩，N64短结果不当正式N256。通过才另行正式4K，不测16K。

最迟2026-10-23 15:58UTC(23:58BJT)停止；代码/报告/关键JSON/CSV推天数GitHub，原始仅容器，当前r3已采用版不变。CPU预备可在质量期间完成，GPU run_pair只有完整双质量门槛通过才允许启动。

v3 changes CPU qualification handling only: require independently verified C8 CPU reconciliation (original GPU run remains audit_failed, no GPU rerun) and normal C64 completion. The GPU model remains the exact944-source v2 model. Record and pin recovery CPU sources separately. All performance thresholds remain unchanged. Leave CPU idle during timed rounds; no hashing, pushes, cache work, or source editing then.

Before the first GPU screen, source review found a second stale NT8 warm-capture assertion in the startup canary controller. Replace it for both arms with the real NN8 exact42-prefix/source/config/capture check. All runtime replay audits remain required. This fixes pre-timing evidence selection only; neither GPU model nor timing work nor threshold changes. Final EXECUTION freeze is regenerated before either arm starts.

Final pre-GPU review also found the same stale NT8 selector in inherited full runtime auditors. Both control and candidate now explicitly select actual NN8 routes, with exact42 prefix-to-existing-bank-ID linkage and frozen source/config checks; existing Graph capture/replay/frame checks remain unchanged. This builds a checked view of actual records and never modifies counters or raw JSON. Old LM precision guards remain intact for r3; W4 replacements remain explicit only in candidate. Full and canary NN8 adaptation frozen before the first screen.
