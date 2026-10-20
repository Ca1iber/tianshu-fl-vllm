# 当前推荐：prefill INT8 正式4K 5586.28 tok/s

新正式4K **5586.28 total /1117.25 output tok/s、TTFT3780.63ms**，新Math **102/105（97.142857%）**。原N256/C64，4轮排首轮，1024成功、零错误/抢占、就绪后无autotune；独立复核采用。相对原默认吞吐+1.89%、TTFT-5.84%，公开基线吞吐+175.46%。本配置没有完整16K，旧3885.91是不同profile。

在原比赛镜像的容器 `/workspace/vllm-plugin-FL` 内运行：

```bash
bash experiments/tianshu_prefill_int8_release_20261021/restore_generated_sources.sh
bash experiments/tianshu_prefill_int8_release_20261021/serve_tianshu_minicpm.sh
```

启动前确认GPU空闲，不与已有评测并行。本脚本Bash/jq直接exec原13参数，逐一核验407运行来源、12主控制来源、62环境；VLLM_DISABLE_COMPILE_CACHE=1。`TIANSHU_PREFILL_INT8_PROFILE_DRY_RUN=1`只核验，不启动服务；`TIANSHU_PREFILL_INT8_PROFILE_RECORD_DIR`可指定记录目录。CPU干运行与实际Math/正式的精确profile绑定；没有为此额外重跑正式测试。

26份生成依赖约1MB与已验证原件字节相同，恢复器仅补缺，已有SHA不符则拒绝覆盖。包不改checkpoint，不含权重、编译缓存或完整回答。新GateUp只在24个预热对齐M做prefill/混合A8W8，M8/M64为NN纯decode，LM8仍BF16、LM64原NT整数，其它矩阵/尺寸保留原路。43 INT8 bank 1326938112bytes真实计入DeviceMemoryProfiler，prefill不新增packed权重payload。

证据：`../tianshu_prefill_int8_official_results_20261021/REPORT_32g.md`、OFFICIAL_GATE/原CSV/summary，`../tianshu_prefill_int8_quality_results_20261021/REPORT_32g.md`和全部105条关键评分/usage/行SHA。完整原件仅容器；另外两个结果CPU复核器不是原冻结12主控制。标签 `tianshu-original-command-4k-prefill-int8-20261021`。
