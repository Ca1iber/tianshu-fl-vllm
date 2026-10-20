# Prefill INT8 候选复现包

本启动器绑定当前正在评测的407份运行来源、12份控制/审计来源、62项环境和原始13项启动参数，逐文件SHA256检查。CPU干运行已通过，不启动GPU。新Math105和正式4K未完成前，本目录不代表新正式成绩或默认推荐。

容器 `/workspace/vllm-plugin-FL` 内依次运行：

```bash
bash experiments/tianshu_prefill_int8_release_20261021/restore_generated_sources.sh
bash experiments/tianshu_prefill_int8_release_20261021/serve_tianshu_minicpm.sh
```

必须先确认 GPU 空闲；正在运行评测时不要重复启动。该包不改checkpoint，不包含模型权重、编译缓存或完整原始评测数据。26份生成依赖约1MB，为旧版本同字节内容；旧版已不再依赖的SCREEN_GATE不打包。恢复器仅补齐缺失文件，已有文件哈希不符则拒绝覆盖。新增实验来源均来自天数仓库。纯Bash启动，可用 `TIANSHU_PREFILL_INT8_PROFILE_DRY_RUN=1` 只验证环境与来源。

候选对齐M128至4096的GateUp大矩阵采用预热后的A8W8；M8/M64纯decode与小测对照相同。小测+1.75%是N64配对结果，不能写成N256正式成绩。质量和正式结果另存独立结果目录后更新。
