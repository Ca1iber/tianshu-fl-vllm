# BI-V150 32G：K8V16 正式4K 4997.05 tok/s

原始官方 serve argv、未修改的官方 benchmark，4096输入/1024输出/C64/N256，
连续四轮、首轮排除。后三轮均值 **4997.05 total tok/s、999.41 output tok/s、
TTFT 4155.54ms**，比上一版双8位4259.87提升 **17.305223%**，
比公开基线2028.01提升 **146.401645%**；TTFT相对上一版4291.83下降3.175568%。

|轮次|Total tok/s|Output tok/s|Mean TTFT ms|成功请求|计入均值|
|---|---:|---:|---:|---|---|
|1|4846.04|969.21|5802.88|256/256|排除|
|2|5001.67|1000.33|4154.79|256/256|计入|
|3|4992.70|998.54|4154.71|256/256|计入|
|4|4996.78|999.36|4157.12|256/256|计入|

1024/1024成功，后三轮吞吐极差/均值0.179506%，零抢占，全部原CSV在下载后独立重算。
首轮日志存在一次 exponential 形状调优：`fused_exponential_kernel_f32_unroll8`，
N=8094720，4个候选及1条汇总，汇总时间0.66s。它出现在第164个完成POST之前，
位于首轮；完整日志之后没有新增调优记录。没有将“后三轮无新增”表述为全过程零调优，
也不据此声称所有JIT成本已消除。原benchmark不保存逐请求响应正文，审核范围为stdout/CSV。

## 实际变化与质量

保持K的固定scale1/16软件有限E4M3编码，让V保持BF16，省去V解码。
真实42层FullAttentionSpec及缓存重塑使用12KiB物理页（双8位为8KiB，原BF16为16KiB），
UInt8 carrier的V区域被视为BF16零拷贝别名，没有额外完整BF16 KV副本。
实际缓存680240 tokens。约512MiB共享prefill arena和131072-byte页表在profile和显存规划内计入；
prefill只展开使用的逻辑页，超容量保留正确但较慢的直接byte fallback。
组件与真实服务都验证了物理页、view别名、KV写入、变数据Graph、输出保护和参考数值。

同配置新Math500 Level3 **100/105=95.238095%**，105成功、0错误、0缺失，
满足官方>95%要求。错误索引18、47、75、87、101；101触及32768上限，仍计错，
104题正常stop。保持temperature1/top_p.95/max_tokens32768、batch8、timeout3600；
没有复用旧102/105或重抽样。单次不同配置的随机生成结果不用于解释题数差异的因果。
完整质量审核见MATH_REPORT_32g.md。

模型/Q/输出仍BF16、attention累加FP32；2048调度预算、max_num_seqs256、async=True、
stream_interval1、75预热点、M2048三个GEMM tile、严格TopP32K几何搜索和原RNG保持原配置。
真实B64 producer为T16/w1/S4，编译记录142寄存器、0spill；旧双8位对应245寄存器。
这是与小规模服务+17.24%一致的正式收益，未以组件或N64分数替代官方N256分数。

## 独立复现与归档

Math启动源0e8d9b8a1256951cbb689f8b21f9a858454b2a1b；正式启动源73536170295dfd6b3a9a40fad5bc147d797096ea。
两次196份冻结源码、原serve argv和实际API的41项环境匹配，仅记录根目录不同。
运行中仅新增报告/独立后续实验准备，目标源码不变；没有修改原官方脚本
`0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`。

`serve_tianshu_minicpm.sh`的实际shell dry-run已与两次完成API的argv/41env分别匹配，
继承的过期实验开关已移除，LAUNCHER_AUDIT_32g.json通过；没有额外用新launcher启动服务。
脚本实启动前会检查全部196个冻结源；仍依赖这些历史记录与源码路径，不是无外部依赖的提交包。

正式包5575277字节、22原分片及整包SHA已核验：
`4a2e7350b0ae09152b9653df855f171e9fa4345f0fccc2ce4d912e44d7e8c569`。
Math包6505917字节、25分片，SHA：
`5402b9ec31e082c2eae23b30adcd81aa33eebe3cfbff42a6081577b1eefb2497`。
正式本地根目录：`service-results/20261019_tianshu/k8v16_engine_official/20261019_k8v16_engine_official4k_tianshu`。
可执行本目录verify_validation_downloaded.py（指定--root及--math-root）和audit_launcher.py复核。
下载中断后只从逐分片SHA已验证的前缀续传，最终仍重新校验完整归档与全部分片。

新检查点tag `tianshu-original-command-4k-k8v16-20261019`，保留旧双8位与GEO tag，
更新最佳记录前完整备份原canonical。采用范围为本次原始命令4K及其新Math；
未跑本版16K，旧16K成绩保留原profile归属。只保存本地和天数容器Git，不额外push。
后续11配置decode sweep和地址32实验尚未运行，不在本版成绩内。
