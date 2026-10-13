# 天数 BI-V150 32g：top-p启动参数三轮4K小收益

在当前正式最好DecodeL0KV32/B75G1/budget2048配置上，仅将实际native top-p启动从w4/stage3改为w16/stage1。4K输入/1K输出、C64/N64，每组3轮预先排除首轮：3468.876674→3528.730902 total tok/s，提升1.725464%。TTFT13359.365619→13310.515118ms，基本持平。当前不采用、不运行新的完整benchmark，先增加同期复测轮数确认。

|轮次|控制tok/s|候选tok/s|控制TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1296.818759|1306.355773|19135.811085|20492.576440|否|
|2|3451.922649|3523.658452|13324.798979|13296.887926|是|
|3|3485.830699|3533.803351|13393.932259|13324.142310|是|

384个吞吐请求全部成功，配对提示种子逐轮一致，usage逐项4096/1024。7项greedy探针一致，但其temperature0不经过随机top-p，不能代替对应质量评测。当前DecodeL0KV32的Math103/105不能复用于新top-p warp配置；若服务收益稳定，必须新独立105项验证。

候选有57组实际top-p路由，编译metadata为w16/stage1，native source SHA64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8。其他6份安装计算源和coherent hook与控制一致，Decode仍TILE32/L0、producer/reducer配对。top-p body、blocks、grid、温度/p/k及RNG保持原函数。微测大部分掩码逐位同原版，ties一位不同但严格CPU语义检查通过；原版及候选normal的严格CPU偏差都保留，详见micro_32g报告，不声明所有输入bitwise或高精度全部通过。

完整原始包3963637bytes、16分片，SHA2567c69a12527e90e8677b096ea509fdfebfe6e7001f2c3a9174f1fc7c9cfc85f61。分片/整包、384请求usage和种子、均值、实际路由及source_snapshot逐字核验；原件目录service-results/20261013_tianshu/topp_launch_service/20261013_topp_launch_service_tianshu。实际三轮源码保存为本检查点，新的五轮确认使用独立文件和目录，不追溯改变旧结果。

当前正式最好仍3426.62/Math103，tag tianshu-best-decodel0-kv32-20261013。新五轮各组排前两轮，用后三轮判断收益，不追加16K。用户已预约2026-10-14整天，新停止时刻为2026-10-14 23:58（北京时间）。未采集硬件性能剖析，TODO。安装源、权重、官方benchmark不改，沐曦只读。
