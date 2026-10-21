# 2026-10-21 20:04北京：逐M分段GateUp融合服务v2

预约至10-22 00:00，GPU硬停10-21 23:58北京（15:58UTC）；0–6不可用。SSH master /tmp/tianshu-20261021-afternoon-control。天数容器77b9f29ce7f8，代码/原始实验都在容器，代码报告关键结果经GitHub保存，禁止host代码/日志副本。

正式最好：5618.01总tok/s、1123.60输出tok/s、TTFT3835.46ms；原命令4K/1K C64 N256四轮排首轮，1024成功，0preemption/0就绪后autotune，热差幅.0854%。官方Math8 101/105、补充真实C64 103/105。release `tianshu_nn_silu_fx_release_20261021/serve.py`；已push tag `tianshu-original-command-4k-nn-silu-fx-20261021`。N64短测不能与此正式数值直接比较。

新方向分离role/M/重排/融合/调度。已完整完成BF16六tile504相位；GateUp宽tile v2全部240配置通过；实际42bank逐M完整MLP v2全部3024图/eager动态输入及24整链poison检查、独立CPU147456样本。按正收益23M选配置，排除M256 -2.334%；M4096完整MLP+9.5348%，多数M+3%~10%。这都是组件，尚非新模型成绩。INT8 bank/精度白名单不扩，新增payload为0。

旧fixed-stream的42bank计时已经撤回，不能引用M4096-7.51%等结论。修正版每次取actual capture stream，2016非零句柄及Graph/eager全输出、poison通过。旧raw保留，withdrawn注释和修正版都已push。新的交错GateUp单dot80配置全部数值合格但性能大幅退化（M64约1.42ms对当前.063ms），不采用。

首次range服务v1控制热5546.14tok/s，候选启动binary检查失败，未测速/未Math/未正式。CPU诊断已证明来自tiny未使用Projected张量导致SME mask19而组件83；安装版get_corex_sme按张量形状分类且JIT会覆盖手传use_sme。v2仅把未使用Projected换为既有2D BF16输出alias，0新增存储，保留每M与组件binary严格匹配。原v1不改源/不重跑。

当前GPU串行pair v2：PID281932；目录 `/workspace/benchmark-results/20261021_m_range_prefill_pair_v2_runner_tianshu`，两臂raw `_pair_v2_control_tianshu`/`_pair_v2_candidate_tianshu`。源码 `tianshu_m_range_prefill_service_v2_20261021`，CPU实际FX+prepare通过471-source/62env/13argv，两臂原450版为一致子集。新动态[1,4096]改写明确审核，旧M64 static fusion两臂保留。不要修改已经GPU运行的source/PLAN/CPUgate。

后续串行验证队列 PID281946，`.../20261021_m_range_prefill_validation_v2_runner_tianshu`。等待pair终态→独立CPU核验→只有后两轮差幅各<1%且吞吐+.5%以上才新Math105 C8→真实C64补充105（原生准入only，fresh source map一致）→原benchmark N256四轮。时间不足不开始下一阶段。控制器新目录 `tianshu_m_range_prefill_validation_v2_20261021`，CPU实际两种prepare通过；Math/C64正式三个raw为 `20261021_m_range_prefill_v2_{math8,c64_quality,official4k}_tianshu`。

GPU未运行的后备：`tianshu_m_range_bf16_large_20261021/probe_gpu.py` 12大M×14tile/group×3role×2phase1008相位；加group1/8和BM256BN256真实M2048 tile，严格保留current O/Down2048 special，用同tile比较W重排；新的CPU核验器 `tianshu_m_range_bf16_large_results_20261021/verify_cpu.py` 已准备。不与service/质量并行占GPU，时间够再跑。

保留用户dirty两份 `tianshu_gateup_silu_20261012`，MuXi只读，不动。Git精确add，不上传大tensor、全responses、snapshot；原始大文件只留容器。GitHub容器直连挂起，用host SSH FIFO stdio bridge推送，不落host内容文件。最新确认push37cc492，后备CPU核验commit2cc7ce4尚待push。每完整4K轮向用户报数，进程timeout先查原PID（Z是终态），不因查询timeout重开GPU实验。
