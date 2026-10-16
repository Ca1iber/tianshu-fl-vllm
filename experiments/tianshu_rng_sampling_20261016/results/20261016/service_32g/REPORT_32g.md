# 天数：Philox融合服务收益0.36794%，暂不采用

新鲜3+3对照，原始官方serve命令，真实默认2048tokens/256seq/asyncTrue/stream1，75个Prefill预热点，现有CV/KV16/w1/B64S8/nativePrefillL0/nativeTopP。只有候选启用新RNG采样融合；两边37项显式环境逐项比对，仅RNG开关0/1和记录目录重映射不同。

4K输入/1K输出/C64/N64，各三轮，预先排除首轮。每轮不同输入种子、跨模式配对，使用temperature1/top_p.95，避免重复N64输入的前缀缓存收益。计分均值 **3938.070561 → 3952.560342 total tok/s，+0.36794111%**；TTFT 13137.524→13195.401ms（+0.44055%）。收益小，不采用，不为单独此项跑新Math或正式4K，不更新3992.97原始命令正式配置。

|模式|轮次|total tok/s|TTFT ms|成功请求|新增调优打印|计分|
|---|---:|---:|---:|---:|---:|---|
|control|1|3744.262999|16825.275|64|0|否|
|control|2|3942.746181|13109.897|64|0|是|
|control|3|3933.394941|13165.151|64|0|是|
|candidate|1|3720.556473|17312.274|64|0|否|
|candidate|2|3954.387226|13200.703|64|0|是|
|candidate|3|3950.733458|13190.099|64|0|是|

采样局部B64 eager0.760865→0.358317ms的降幅约53%，进入整模型只有约0.37%吞吐提升。局部计时不代表整体性能；没有将噪声张量带宽收益外推成模型收益。新全局Philox随机序列与旧exponential_不同，保留独立API generator回退，不借旧数学准确率。

384份完整生成文本/usage/输入种子/TTFT与耗时逐一复核；每轮64成功、usage4096/1024/5120，6轮源和原始请求配置一致，实际37环境及52源快照已本地重算。4个API seed777 canary自身及跨模式文本一致；候选有B64融合与seeded fallback路由记录。路由记录是每batch/route首次观测，可能包含启动dummy sampler，未声称记录了每个运行期采样调用。SDK和hook代码契约、独立API种子回退及整个服务A/B结果分别保留。16份metrics快照均零preemption。每轮新增调优打印为0条，不等于全面测量所有编译成本。

前置V2微测105数值/12整数RNG/状态/capture/频率检查通过，详见相邻micro_v2_32g；首版NaN参考契约差异的失败及诊断保留在v1_boundary_failure_32g。所有FP64参考在CPU，不引入模型量化、服务参数修改或官方benchmark脚本修改。

原始包5761996bytes/22parts，SHA `52a8dd3d2b3530de6338820308f99bc616e1ed320f145dd0317b0e9113b066b3`，分片与整包已核验；本机raw为service-results/20261016_tianshu/rng_sampler_service/20261016_rng_sampler_service_tianshu。离线复核：

```bash
python3 experiments/tianshu_rng_sampling_20261016/verify_service_downloaded.py --root <ServiceRaw> --micro-root <V2MicroRaw>
```

当前两边服务均已退出，GPU空闲。未运行新Math、完整4K或16K，本报告不是正式成绩。后续可将此小收益作为组合候选，但不能假定不同优化的收益可相加，也不能跳过新组合质量验证。用户原WIP保留，沐曦库只读，无push。
