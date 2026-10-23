# Down64 W4 v2正式4K：吞吐+0.6149%，TTFT未达门槛，未采用

原始4096/1024/C64/N256，4轮排除首轮。total tok/s依次5771.42（排除）、5985.30、5992.38、5984.34；后三轮官方summary为5987.34 total /1197.47 output tok/s，TTFT3240.21ms，吞吐极差/均值0.134283%。同场N64短测增0.562771%，不是正式分数。

相对历史r3 5950.75 total、TTFT3174.58ms，吞吐+0.614880%，TTFT+2.067361%。后三轮TTFT分别3221.48/3347.58/3151.58ms。固定晋级要求吞吐>0.5%、热轮spread<1%、TTFT<=r3*1.01；前两项通过，TTFT未通过，因此promotion_passed=false、adopted=false，发布草稿不构建、不启动、不打合格标签，正式保留r3。

限制：本轮与历史r3不是同场完整AB，第2/3轮期间还进行了另一候选的CPU源码/文件校验准备，没有CPU资源隔离；这可能影响TTFT，不能把全部延迟变化归因于W4。后续计时期间停止后台CPU准备。本结果保留，不原样重跑追分。

独立CPU完整重算本版916来源及控制器和全部快照、原始官方脚本SHA、全部1024成功请求、完整质量raw SHA、所有旧GU/KV/scheduler/sampling/LM/attention/其他M Down审计与新42W4捕获/模型重放/实际load及native memory scratch/accounting。零就绪后Triton autotune、零抢占、退出后GPU空闲，checkpoint SHA不变。新鲜C8 102/105、C64 103/105，全部cap计入分母。质量通过不代表本轮性能晋级。

OFFICIAL_GATE和KEY_RUNS是关键核验与成绩，CPU_COMPARISON.json.gz为压缩完整审计，原始全部位于/workspace/benchmark-results/20261023_dots_Down64_v2_official4k_tianshu。没有16K结果；MuXi仓库未修改。
