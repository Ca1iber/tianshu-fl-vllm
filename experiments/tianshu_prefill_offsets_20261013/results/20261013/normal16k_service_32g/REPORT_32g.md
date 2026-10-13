# 天数32g：正常16K/1K输出下的Prefill stage2同期A/B

恢复16K输入/1K输出、C64/N64，每组3轮排首轮，其他配置保持Decode地址32、B75G1、Native L0 M256/KV128/w16、budget2048。唯一变量Prefill pipeline1→2。计分均值2562.827466→2624.979801 total tok/s，+2.425147%。不是原始N128正式分数。

|轮次|stage1 tok/s|stage2 tok/s|计分|
|---:|---:|---:|---|
|1|2254.431804|2305.686244|否|
|2|2563.553302|2625.559823|是|
|3|2562.101630|2624.399779|是|

计分TTFT 154893.872563→149703.542118 ms，变化-3.350895%。六轮384请求全部成功、usage每条16384/1024，配对种子与7greedy响应相同，安装源与Decode32及其它hook SHA一致。真实Prefill stage2 configuration/router引用及模型route均已核验，非空环境设置未生效的伪对照。

8场景96CPU/GPU FP32/Graph/完整输出逐位相同先行通过。16K/128 C32/N32的筛选+5.39%没有直接当作1K输出增益，本次确认正常输出长度下+2.425%。候选两个计分轮都高于控制最高轮，满足后续质量与正式验证的收益阈值。新的独立stage2 Math已自动启动；旧profile102/105没有复用给新stage2，>95%后才运行原始16K/N128及4K/N256。

全部raw4224411bytes/17分片，整包SHA256 1d8e0f4543ae19aa981453f585e977dfdc771ba33b782b74435a2b8d935da193与全部分片已本地verified。目录service-results/20261013_tianshu/prefill_stage2_normal16k_service，实际harness、override、控制器、全部响应与路由保留。安装文件与原始benchmark未改。
