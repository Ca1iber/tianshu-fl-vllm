# 天数软件8位KV：正式4K 4259.87 tok/s，较前版提升4.06%

2026-10-19 18:02本地完成。原始官方serve命令、未修改benchmark脚本，4096输入/1024输出/C64/N256，连续4轮，首轮排除。后三轮均值 **4259.87 total tok/s、851.97 output tok/s、平均TTFT 4291.83ms**。相对此前同原命令最佳4093.81提升4.056368%，相对公开2028.01提升110.051726%；TTFT比前版4309.35下降0.406558%，通过公开TTFT约束。

|轮次|Total tok/s|Output tok/s|Mean TTFT ms|成功请求|计入均值|
|---|---:|---:|---:|---|---|
|1|4163.02|832.6|5786.6|256/256|排除|
|2|4258.89|851.78|4296.97|256/256|计入|
|3|4261.76|852.35|4287.3|256/256|计入|
|4|4258.95|851.79|4291.22|256/256|计入|

共1024/1024成功，后三轮吞吐极差/均值0.067373%。完整server.log未见autotuning记录，前后metrics零抢占；未据此声称不存在任何JIT成本。原脚本SHA256 `0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`；原脚本保存stdout/CSV，不保存逐请求响应正文。

## 实现与取舍

真实WorkerFL/ModelRunnerFL生命周期将42层FullAttentionSpec和实际KV views改为UInt8，每个page从16384减为8192 bytes；采用固定scale=1/16的软件有限E4M3编码，不依赖硬件FP8。模型权重、Q和输出保持BF16，attention累加FP32。它会改变量化后KV数值，不能使用旧版正确率作为证明。

直接byte decode按实际batch/长度选择分段；大batch 2D使用BM16/BQ2/T16/w1，组件记录245寄存器、0spill，避免旧T128/w16大图捕获OOM。prefill只解码当前使用的逻辑页到共享BF16 arena后调用原生attention，避免全部prefill直接解码带来的高spill；mixed batch分别处理decode与prefill。约512MiB共享arena及131072-byte页表在真实profile_run/内存规划内预留计入，只复用一个buffer，不暗藏全量BF16 KV副本。超出有界arena时保留正确的直接byte fallback，组件已测，该罕见路径仍可能较慢。

保留2048调度预算、max_num_seqs256、async=True、stream_interval1、75个预热点、M2048三个投影tile、strict TopP32K几何搜索和原采样RNG；原attention/gather索引宽度不变。此工作负载的收益来自KV存储与attention路由，不能推断其他长度收益。

## 独立质量与配置复现

新的Math500 Level3 **102/105 = 97.142857%**，105成功、0错误、0 incomplete，超过>95%要求；全部105条预测及105条review独立重算。错误索引[18, 87, 72]；stop104、max_tokens1（索引83，仍被判正确）。保持temperature1/top_p.95/max_tokens32768、batch8、timeout3600。没有复用此前同为102/105的旧输出。

正式运行来源提交 `9f7d922d8576697c09fbdfa75054fb398db33d89`，Math来源 `9c0d9ed352b2a42755fe950fee071091fc5ce5fc`。两次相同159份源码SHA、原serve argv、实际/proc观测40项环境均一致（仅记录根目录不同）。`serve_tianshu_minicpm.sh` + `serve_profile_template.json` 提供相同profile；真实shell dry-run与两次完成API的argv/40env均匹配，注入旧开关已移除。新launcher没有额外实际启动，成绩由相同profile的原验证控制器取得。实启动需历史冻结源/记录引用仍可读，脚本逐一校验159份SHA；这不是不带依赖的提交包。

正式原件：`service-results/20261019_tianshu/soft8_engine_v2_official/20261019_soft8_engine_v2_official4k_tianshu`，5370105 bytes，21分片及整包已校验，SHA256 `a0843e215cf379b50224d51a09f7afb4014572cca3adbcf27e2af73917ae8cdb`。新Math包SHA256 `aed08091d275dd9a885df779d09a4ce71d5310b9116dd42bc72db2d3730d586f`。原CSV、runtime路由/内存记录、159份源快照、实际环境及质量原件均本地独立核验。

```bash
python3 experiments/tianshu_soft8_engine_v2_20261019/verify_validation_downloaded.py \
  --root ../service-results/20261019_tianshu/soft8_engine_v2_official/20261019_soft8_engine_v2_official4k_tianshu \
  --math-root ../service-results/20261019_tianshu/soft8_engine_v2_math/20261019_soft8_engine_v2_math_tianshu
python3 experiments/tianshu_soft8_engine_v2_20261019/audit_launcher.py \
  --math-root ../service-results/20261019_tianshu/soft8_engine_v2_math/20261019_soft8_engine_v2_math_tianshu \
  --official-root ../service-results/20261019_tianshu/soft8_engine_v2_official/20261019_soft8_engine_v2_official4k_tianshu
bash experiments/tianshu_soft8_engine_v2_20261019/serve_tianshu_minicpm.sh
```

本版采用范围是已验证的原始命令4K工作负载；未跑本版16K，不宣称完整比赛资格。新检查点tag `tianshu-original-command-4k-soft8-20261019`，旧geo最佳tag保持。后续binary32解码只是另一项待测实验，不在本次成绩中；本检查点仅保存本机和天数容器，不额外push。
