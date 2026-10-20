# 长上下文S16候选的同配置启动入口：16K正式均值待完成

已完成新Math Level3 102/105（97.142857%）、原始正式4K四轮5099.65 total tok/s；原始正式16K四轮正在运行，当前不发布新的最佳标签，也不把N64小测3751.55当正式成绩。

serve_tianshu_minicpm.sh使用与新Math及当前原始双负载相同的46项环境和225份冻结源，校验源码SHA后执行原始serve argv；脚本本身不宣称分数。它只启用已测K8V16/4096/83预热/严格TopP32K/M2048及长上下文S16，不包含等待中的KV水位实验。DRY_RUN_GATE_32g.json记录CPU dry-run与完整46环境归一化比对，无模型执行或GPU性能宣称。

必须在天数同版本环境运行；需要的历史22个生成代码路径可通过[恢复入口](../tianshu_k8v16_restore_inputs_20261020/README_32g.md)补全，再使用本目录的serve_tianshu_minicpm.sh。新克隆整套环境安装/实际启动尚未验证；当前容器的全部225份源已核验。原始模型权重、数据集、缓存和日志不上传GitHub。

证据：[新Math](../tianshu_long_segments_quality_20261020/REPORT_32g.md)、[原始正式双负载进度](../tianshu_long_segments_official_results_20261020/README_32g.md)、[同负载16K服务对照](../tianshu_long_segments_results_20261020/REPORT_16K_32g.md)。
