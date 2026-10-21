2026-10-21T16:28:22+08:00

# 当前继续点

预约到2026-10-22 00:00北京时间，GPU硬停止2026-10-21 23:58（15:58 UTC）。活动目标继续优化，不暂停或宣称完成。新SSH master是本地主机 `/tmp/tianshu-20261021-afternoon-control`，10h持久连接；旧master约16:27过期，不再依赖它。

活动GPU任务：runner PID244330，`/workspace/benchmark-results/20261021_nn_silu_fx_screen_runner_tianshu/phase.txt`。必须先ps核实，不因观察超时重跑。同次任务依次control→candidate→reference，4K/1K、C64/N64、各3轮剔首轮。冻结入口commit5a58a3d已push。数值部署复用0afa16c的原生FX融合；本新实验加入精确index启动预热，13 argv/62 env、原始407源子集保持。

control完整通过：5281.9952/5532.1350/5526.8035 total tok/s，TTFT12520.24/10326.83/10410.19 ms；后两轮均值5529.4693，spread0.0964193%，无调优、完整运行审核通过。candidate已启动尚无轮次。reference是原始407最佳的真正参考，不含静态compile64或新index钩子。结果根均为 `/workspace/benchmark-results/20261021_nn_silu_fx_screen_{control,candidate,reference}_tianshu`。

最新正式最好仍tag tianshu-original-command-4k-prefill-int8-20261021 /521c1541c2d2763eddc72ed43e7d26cb40b638fa：5586.28 total /1117.25 output tok/s，TTFT3780.63ms，Math105=102/105，407源/62环境/13原始参数。当前实验不复用其准确率，不称为正式分数；小测与N256不可直接混比。

本轮证据：双dot组件M64含量化0.0855160→0.0629089ms（速度+35.94%，时间-26.44%）；16配置/48阶段严格逐位及CPU复核。single原版整数错误已拒绝，join实现正确但约0.278ms更慢。报告与全部关键gate已推送8a4e792。

模型只读FX观察c06de0e/b213623：86图覆盖42层的静态64和动态[1,4096]；实际生成代码消除了SiLU中间BF16 cast，使用FP32 x*sigmoid(x)*y。组件fdiv形式算术顺序仍有差异，须新鲜质量，不能宣称模型逐位。

首次接入0afa16c真实完成3冷C1+64并发128/128，42层融合二进制与组件c42a5346...完全相同（83regs/0spill）；M64图126次真实replay，43动态原图不变，M8及24prefill暖尺寸/42真实请求路径均保留，43权重签名保持，冷C1token前后一致。但请求期有index M4001/4032及M32两条调优，原始state/result保持failed。08298d1的独立CPU恢复验证其余结构，不伪造原试验通过。后续新5a58a3d启动预热input长度63/64×index32/4001，保留RNG并逐项CPU对照，服务期不拦截index。

接下来：观察每个4K轮次并报吞吐/TTFT。三臂完毕CPU verify_screen会核对576响应、原始407参考、全部源码/完整运行审核并写SCREEN_GATE/ROUNDS/CSV。相对原始参考>0.5%且稳定才列候选；新鲜官方Math严格>95%，还要补实际覆盖M64融合的质量检查（官方batch8主要走M8），通过才做正式N256，不例行16K。失败后不改冻结代码；修复用新目录。运行审计若有数据后处理问题，只CPU恢复，不重跑GPU补元数据。

CPU预检中两次目录的记录路径归一化已在GPU前修正；旧CPU失败日志保留。FX结构168检查全部通过；屏测CPU公平性检查450源通过，两个新增index形状启动移前有明确代码变化，不靠已有缓存重跑冒充修复。

其他只读调查：V8线性和magic组件均已负收益（+32%/+46%耗时），不重复。采样logits-log(noise)旧组件105检查通过，但旧4K/128服务仅+0.217%；旧栈实际batch256，不能当作当前4K/1K效果。新的采样接入或K6压缩尚未实现，不误报为进展。

用户修改的experiments/tianshu_gateup_silu_20261012/{experimental_fusion.py,probe_fusion.py}必须保留。所有代码/报告/关键数据只在容器和天数GitHub，无本地副本。不改沐曦仓库。不用Python包装常规shell/git。Git精确暂存、检查空index后commit/push，skip ci。
