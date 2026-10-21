# 天数逐M融合：Bash复现入口（等待正式晋升）

在天数原比赛镜像中使用；仓库为/workspace/vllm-plugin-FL，模型为/workspace/MiniCPM5-2B。
官方C8新题103/105由CPU单独复核，原GPU覆盖审核失败保持；真实C64新融合质量101/105已通过。
当前正式4K仍在运行，尚未生成QUALIFIED_RELEASE.sha256，所以此入口暂不允许启动服务。

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_m_range_prefill_release_20261021/restore_generated_sources.sh --check
TIANSHU_RANGE_FUSION_PROFILE_DRY_RUN=1 bash experiments/tianshu_m_range_prefill_release_20261021/serve_tianshu_minicpm.sh
```

晋升后，直接Bash启动原13参数vllm服务：

```bash
bash experiments/tianshu_m_range_prefill_release_20261021/serve_tianshu_minicpm.sh
```

每次启动验证471模型源和独立控制器、62个实际环境值；正式晋升条件及两轮质量证据绑定后才执行vllm。
未知M/未选中M保持原路径，M256明确排除；复用已有43个packed bank，没有新增权重payload。
--check只校验已归档的26个历史生成输入；不带--check时仅补充缺失且hash匹配的文件，已有不同内容会拒绝覆盖。
生成源码已有Git副本，完整原始评测回答/张量保留容器。此入口尚未改变原5618.01正式推荐。
