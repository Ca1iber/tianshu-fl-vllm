# 2026-10-21 22:23：BM8整模型待验证

正式推荐版仍为471源M-range版：5695.95 total、1139.19 output tok/s，TTFT3663.77ms，正式N256四轮排首轮，Math8原103/105经独立CPU覆盖审计、实际C64 Math101/105。tag `tianshu-original-command-4k-m-range-fused-20261021`、0291481已在GitHub。本轮未重跑16K。

QKV M64 NN组件+19.85%但新同次整模型5625.989→5641.429，仅+.2744%，未达>.5%门槛；0抢占/0autotune，全384响应独立审核，42层捕获和实际C64回放已核验。新增220631040字节packed权重/实际KV块40133→39827，不采用，无新Math/N256。报告在`tianshu_qkv_nn_decode_service_results_v2_20261021`。

现行471版新阶段诊断完成：40步实际C64/FULL Graph，总流区间36.0305ms、forward31.8807/logits.610917/sample3.39867；分别88.4806/1.6942/9.4406%。481观察来源、66真实配置、128条4K/512及8条C8 smoke、seed文本复现、权重和主要实际路径审核，全部窗口独立CPU重算通过。区间含提交间隙，不是kernel活动时间或正式吞吐。报告在`tianshu_current_stage_event_results_20261021`。

新K8V16 BM8组件四臂独立对照：current、BM8、BM16_direct、BM8_direct。14×2×4=112全输出、8044544元素均逐位相同，304 CPU GQA向量独立重算、同Graph修改Q/K/V/table/lengths及capture1→4096/8193全通过。C64 4K .552524→.528048ms(+4.635%)、5K .682641→.653482(+4.462%)，实际grid64×2×4、BM8/BQ1/T16/W1/S1、167reg/0spill。BM8单独负收益，BM16_direct约+.5%；只有组合值得模型验证。报告和门槛在`tianshu_k8v16_bm8_v2_20261021`。

本次新直接471 reference已complete：5400.14456(排除)、5611.22995、5621.61142，热均5616.42068，差幅.184841%。raw `/workspace/benchmark-results/20261021_BM8_direct_reference_tianshu`。
新service_v1在未注册旧pair模块的class来源inspect处失败，未运行kernel。保留源码/原件SHA。service_v2仅改方法来源检查，CPU真实未注册模块回归通过。新497源候选仅实际3D/M64启用BM8+直接映射，其余批次/2D调用原JIT；无新packed权重。对照客户端与现行原版字节相同，原13argv/62环境归一化仅PYTHONPATH不同。
当前service_v2 runner PID308178、API308194、EngineCore308335正在启动；raw`20261021_BM8_candidate_v2_tianshu`。实际新源/grid/编译binary需与两组已测组件binary精确匹配，42层捕获和实际C64回放单独审核；旧pair记录仅代表子选择器前参数和基础源，不能误作最终kernel。
quality_v2 runner PID308670等待候选screen CPU gate，稳定>.5%才新官方C8 Math105、实际C64 Math105、官方4K N256四轮。新Math不复用旧分数，不重采样追分。相同497源数值profile通过CPU准备；C64只改admission，实际新路径和原MLP/LM/分段融合均需验证。
所有新代码/报告/关键门槛应推GitHub；原始日志、响应、张量只在容器，本地无项目副本。GPU硬停2026-10-21 23:58北京=15:58 UTC，预约结束10-22 00:00。继续用afternoon SSH master socket(0bytes)，不新建本地项目目录。
