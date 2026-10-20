# 已验证同配置入口：4K 5099.65、16K 3723.12 total tok/s

本版本已完成全新Math Level 3 102/105（97.142857%），以及原始正式4K和16K各四轮；预设排首轮后均值分别为5099.65、3723.12 total tok/s，完整1536/1536成功。16K较上一正式版3213.11提高15.87278%，4K较5101.45变化-0.03528%，基本持平。标签`tianshu-original-command-4k16k-k8v16-longseg-20261020`，详见[完整正式报告](../tianshu_long_segments_official_results_20261020/REPORT_32g.md)。前后正式分数为历史比较；严格同负载N64对照另见报告。

serve_tianshu_minicpm.sh使用与新Math及原始双负载相同的46项环境和225份冻结源，实际启动前校验源码SHA，然后执行原始serve argv。它启用已验证K8V16/4096预算/83预热点/严格TopP32K/M2048及长上下文S16；不包含独立实验中的KV watermark和packed-load候选。DRY_RUN_GATE_32g.json记录CPU dry-run与完整46项环境归一化比对，无额外GPU执行。

在天数同版本环境执行`bash experiments/tianshu_k8v16_longseg_release_20261020/serve_tianshu_minicpm.sh`。需要的历史22个生成代码路径可通过[恢复入口](../tianshu_k8v16_restore_inputs_20261020/README_32g.md)补全。新克隆整套环境安装/实际启动尚未验证；当前容器全部225份源已核验。模型权重、数据集、缓存和日志留容器。

证据：[新Math](../tianshu_long_segments_quality_20261020/REPORT_32g.md)、[正式双负载](../tianshu_long_segments_official_results_20261020/REPORT_32g.md)、[采用判定](../tianshu_long_segments_official_results_20261020/ACCEPTANCE_GATE_32g.json)、[同负载16K对照](../tianshu_long_segments_results_20261020/REPORT_16K_32g.md)。
