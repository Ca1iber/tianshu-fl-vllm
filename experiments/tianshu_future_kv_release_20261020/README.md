# 天数未来KV准入版本：启动与复现

最终采用状态和正式4K/16K成绩见../tianshu_future_kv_official_results_20261020/OFFICIAL_GATE_32g.json；该文件尚未生成时，本目录仅代表已通过CPU启动演练的候选。新Math为103/105（98.0952%）。本版本包含小暂存区128MiB、max_num_seqs64和未来输出KV准入，保留K软件8/V BF16、长上下文S16、4096预算、83预热点、原生采样与stream_interval1。RoPE/KV融合和CTA顺序试验均未纳入。

在比赛容器运行：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_future_kv_release_20261020/restore_generated_sources.sh --check
bash experiments/tianshu_future_kv_release_20261020/serve_tianshu_minicpm.sh
```

入口用shell/jq直接导出52项测量环境，校验261份运行来源，exec原始vllm serve argv，不经Python启动包装。它会清理继承的实验开关，保留其他常规环境。默认将记录写到/workspace/benchmark-results/future_kv_profile_时间_PID；可用TIANSHU_FUTURE_KV_PROFILE_RECORD_DIR指定。原始模型/驱动/依赖包仍须位于比赛镜像相应路径；改变镜像不等同已验证环境。

若容器重建后临时生成源码缺失，去掉--check运行restore_generated_sources.sh再启动。它只恢复清单中22份Python代码，已有文件必须SHA匹配，冲突即退出；不会覆盖不匹配文件。代码包共771219 bytes，不包含模型、编译缓存、日志或完整回复。这样GitHub保有原始来源校验所需的生成代码，测试原件仍留容器。

仅检查、不启动GPU：

```bash
TIANSHU_FUTURE_KV_PROFILE_DRY_RUN=1 bash experiments/tianshu_future_kv_release_20261020/serve_tianshu_minicpm.sh
```

DRY_RUN_GATE_32g.json记录了实际261份SHA检查、原始13项argv、52项实际导出环境、污染开关清理和带空格输出路径验证。它只证明启动配置，没有新服务或性能测量。正式运行与质量证据另存，不将新启动脚本本身当作额外一次GPU验证。
