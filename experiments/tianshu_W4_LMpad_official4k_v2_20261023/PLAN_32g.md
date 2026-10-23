# LM1..64 W4新模型原始正式4K，强制双质量与同期短测

质量进行中提前冻结协议：same944-source模型、13项原serve参数、完整63环境；fresh105题C8/C64均>95%且独立CPU复核，r3/新候选同期N64x3排首轮gain>.5%、候选spread<1%、TTFT<=control*1.01后，才允许正式GPU。CPU准备可先完成，但GPU入口及profile.prepare均执行全部门槛，未满足禁止启动。

正式保持4096输入/1024输出/C64/N256，原官方benchmark脚本四轮、排首轮。全部1024请求必须成功，usage/种子/长度及CSV独立重算，0preempt/无ready后Triton autotune，所有新旧runtime/weight/native memory profiler/source/controller/env/graph检查保留。CPU在全部正式计时轮空闲，仅轻量poll，逐完整轮汇报。

旧合格r3正式5950.75 total/1190.15 output tok/s、TTFT3174.58ms为历史比较（非同期AB），采用门槛固定gain>.5%、热轮spread<1%、TTFT<=3174.58*1.01。短测另提供同期对照，二者不能混称。组件成绩与N64都不是本正式N256结果；未过全部质量和性能不采用，不原样重跑挑结果。当前原r3发布不改。

LM8旧BF16和LM64旧W8要求已显式换成真实新W4调用，Down64 42真实W4捕获联系实际B64 replay；其他GU/QKV/O/KV/attention/scheduler/sampling及native admission检查保持。新增W4载荷不当净显存节省，真实峰值含pad/native工作区和原副本。

不测16K，最迟2026-10-23 15:58UTC停止。代码/报告/关键结果推天数GitHub，完整原始只容器。只完整独立正式门禁通过才可构建新发布、打tag，不能把本预备控制器当新成绩。
