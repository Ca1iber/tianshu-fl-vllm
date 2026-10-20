# 5482.79正式4K / Math101版本的直接Bash入口

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_lm_precision_release_20261021/restore_generated_sources.sh --check
bash experiments/tianshu_lm_precision_release_20261021/serve_tianshu_minicpm.sh
```

容器重建缺少生成代码时，先运行不带--check的restore。已有不同SHA的文件不会被覆盖。入口直接清理继承的profile flags、导出实际测量的62环境，检查359来源与10控制文件，exec原13项serve参数。需要原比赛镜像、原模型和依赖；FlagGems仓库保持4cdc3b760205bb4ce7edd9099fa19fa4986adc24。

CPU检查：`TIANSHU_LM_BF16_PROFILE_DRY_RUN=1 bash experiments/tianshu_lm_precision_release_20261021/serve_tianshu_minicpm.sh`。已用故意冲突的继承LM模式和编译缓存开关验证重置正确，检查不启动GPU服务。入口本身不是独立GPU benchmark，测量证据来自同配置真实服务。

小批次LM BF16、LM64整数、GateUp8/64整数；BF16预填充linear_contiguous，gpu4静态bank加载，Future-KV、成对K读取和分段attention沿用原配置。原BF16 GPU权重保留，额外bank由原checkpoint重新生成并计入真实内存profiling。

生成代码包只有26份.py和1份关键SCREEN_GATE JSON，合计1003331bytes。不包含权重、回答、compiler cache、raw tensor或完整日志。此版无完整16K，不将此前不同配置3885.91归给本版。
