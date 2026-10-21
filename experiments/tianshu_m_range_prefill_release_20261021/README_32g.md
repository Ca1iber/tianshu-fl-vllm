# 天数逐M融合：正式4K推荐版

正式原N256四轮排首轮：**5695.95 total / 1139.19 output tok/s，TTFT3663.77ms**。
较旧5618.01吞吐+1.3873%，TTFT较3835.46ms降低4.4764%；热轮差幅0.30495%，1024请求全成功、0抢占、就绪后日志无autotune记录。
公开baseline2028.01，总吞吐+180.864%=2.80864倍。

官方C8全105题103分已独立CPU复核；其原GPU实验因新融合实际0层而audit_failed，保持原记录。
新C64全105题101分=96.1905%，真实M128/768/4096×42层融合、M64旧融合图重放与LM64 INT8均通过。
正式结果与两轮质量绑定同471模型源、62配置和13个官方serve参数；C64补验只改变native admission配置。

在天数原比赛镜像中使用，仓库/workspace/vllm-plugin-FL、模型/workspace/MiniCPM5-2B：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_m_range_prefill_release_20261021/restore_generated_sources.sh --check
TIANSHU_RANGE_FUSION_PROFILE_DRY_RUN=1 bash experiments/tianshu_m_range_prefill_release_20261021/serve_tianshu_minicpm.sh
bash experiments/tianshu_m_range_prefill_release_20261021/serve_tianshu_minicpm.sh
```

入口直接用Bash执行vllm。启动前校验正式晋升门禁及其hash、471源和冻结控制器、62个实际环境值。
23个正收益M使用预选GateUp融合；M256、未知M保持原路径，复用现有43个packed bank，没有新增权重payload。
历史26个生成输入已有Git副本；restore --check只校验，不带--check仅补缺失且hash匹配文件。

成绩与逐轮CSV见../tianshu_m_range_prefill_official_results_20261021/REPORT_32g.md；原始回答、张量、全部日志保留容器。
本版没有重新测16K，不能引用旧版16K成绩作为本版结果。标签tianshu-original-command-4k-m-range-fused-20261021。
