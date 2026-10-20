# Selective W8A8 original-command full4K

2026-10-20，BI-V15032GB，原始benchmark脚本SHA0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565，原13项serve argv。4096输入/1024输出、C64、N256，4轮排首轮，正式5375.50 total tok/s、1075.10 output tok/s、平均TTFT4191.14 ms。

|轮次|total tok/s|output tok/s|TTFT ms|请求|计分|
|---|---:|---:|---:|---:|---|
|1|5234.09|1046.82|5809.85|256/256|排除|
|2|5365.18|1073.04|4253.39|256/256|是|
|3|5380.04|1076.01|4160.37|256/256|是|
|4|5381.29|1076.26|4159.65|256/256|是|

独立按原CSV Decimal重算三轮均值、核对原stdout四组值与summary一致。1024/1024成功，零抢占，所有Future-KV虚拟预留释放。计分吞吐极差/均值0.2996929%。较上一版成对K正式5235.52提升2.67366%，较公开2028.01提升165.0628%；TTFT较上一版4161.08增加约0.72%。独立门槛通过并采用为新4K版本，原comparison.json的adopted=false历史值未改写，采用决定见OFFICIAL_GATE_32g.json。

新MATH-500同105题、C8、temperature1/top_p0.95/max_tokens32768：100/105=95.238095%，严格大于95%，无API错误，103个stop、2个max_tokens。没有复用旧103/105。全部43组原BF16、INT8、scale哈希在成功小规模screen、新Math、完整4K之间一致；实际C8/C64的42个GateUp捕获与模型回放、Graph外LM head整数调用均核验。

实际 loader-time打包42个GateUp与1个ParallelLMHead，embedding和原BF16 checkpoint未修改，所有原BF16权重保留。新增1326938112bytes已在实际活跃的FL DeviceMemoryProfiler内计入模型占用；真实KV40731blocks。冻结311份运行来源、9份控制器、57项实际进程环境，统一VLLM_DISABLE_COMPILE_CACHE=1防止旧AOT绕过新分支。运行时FlagGems仓库commit4cdc3b760205bb4ce7edd9099fa19fa4986adc24。

复现：bash experiments/tianshu_dense_w8a8_release_20261020/serve_tianshu_minicpm.sh。该Bash/jq入口CPU核验来源/环境后直接exec原参数；启动器本身未单独GPU复测。模型文件和依赖仍需保留；26个小型生成Python来源随代码包恢复，完整响应、模型和编译缓存不进GitHub。

本新profile只完成完整4K，未跑完整16K。旧3885.91是另一Future-KV profile，不能合并为本版全场景成绩。权重打包GPU化和SQL目标表反射仍是独立后续实验，未包含在本次5375.50成绩中。

原始目录：/workspace/benchmark-results/20261020_dense_w8a8_official4k_tianshu。新质量目录：/workspace/benchmark-results/20261020_dense_w8a8_math_tianshu。Git保留关键CSV/JSON/报告，完整原始资料保留容器。
