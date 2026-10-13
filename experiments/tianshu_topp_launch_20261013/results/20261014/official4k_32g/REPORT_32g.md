# 天数 BI-V150 32g：TopP启动参数正式4K结果

当前Decode L0/KV tile32/B75G1/budget2048上，仅叠加native TopP w16/stage1。官方原脚本4K输入/1K输出、C64/N256四轮，预先排除首轮；正式均值3506.84 total tok/s、701.37 output tok/s、TTFT 4494.12ms。相对前次正式3426.62提升2.341082%，相对官方基线2028.01提升72.920252%。同期小测五轮排前两轮提升2.181653%，单独记录，不用小规模3533.78作为正式成绩。

|轮次|总tok/s|输出tok/s|TTFT/ms|duration/s|计分|
|---:|---:|---:|---:|---:|---|
|1|2179.46|435.89|7030.88|601.4|否|
|2|3374.46|674.89|4491.9|388.42|是|
|3|3574.57|714.91|4494.1|366.68|是|
|4|3571.49|714.3|4496.37|367.0|是|

1024/1024请求成功，0失败；逐项usage总输入1048576、总输出262144。新独立Math-500 Level 3为103/105（98.095238%），105完整预测/判分、0错误、全部自然stop；准确率严格>95%。对应新TopP质量包SHA b1b7e4fb48794586d590dac564da2fdd56e09ca93d70300c86d7a98c916a2af7，已本地核验。该质量重新执行，未复用旧版本同为103的结果。

后三轮极差/均值5.706277%；仍有逐轮差异，不把完整成绩写成无波动。首轮尾部变慢时最近250条日志未见调优/编译信息，这不足以证明Graph内耗时覆盖或断言根因。后续独立能力微测检查external event及RMSNorm，尚不混入本轮分数。

实际57组TopP路由compiled metadata=w16/stage1，native源SHA64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8；函数体、grid、blocks、RNG、温度和p/k不变。六份安装源、server argv、TopP hook与新质量快照匹配，64组Decode producer/reducer tile32配对，L0 generated SHA4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e。启动75点预热与int32 gather实际记录核验通过。KV32指tile32 tokens，整数地址偏移保持64位；无量化、无投机解码。

原始包2529027bytes、10分片，SHA256 3e6525f4c5d278a8e86aed803530309f831d51bc9f32fc6e93fbf6803075318e；分片与整包、CSV四行、均值、所有计算源和9份controller冻结源码均已本地验证。原件service-results/20261013_tianshu/topp_launch_official4k/official4096。原始benchmark SHA0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565，未修改。controller正确串行完成math和official4096两个job，退出码均0。

结论：正式吞吐提高，可作为新的4K最好配置；保留旧tag可回退。 本报告只覆盖4K，新配置未运行正式16K。性能剖析：TODO，尚未采集有效Graph内部覆盖。安装源、权重、采样与官方benchmark原件不改，沐曦参考仓库只读。
