# 天数 BI-V150 32g：当前最好版本缩小prefill预算为负优化

保持TopP16/coherentDecode32/nativePrefill/G1及有界bucket策略，只改全局max_num_batched_tokens2048→1024。4K/1K C64/N64各3轮，预排首轮：3533.492990→3241.147956 total tok/s，变化-8.273542%。384请求全部成功；性能下降，不采用、不跑该版本新Math或正式benchmark。

|轮次|2048tok/s|1024tok/s|2048 TTFT ms|1024 TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1288.644150|1157.944514|19007.866667|39010.878441|否|
|2|3525.534225|3134.805991|13261.089640|16802.364511|是|
|3|3541.451754|3347.489920|13259.318566|16820.755684|是|

实际控制75点/75次预热，候选67点/67次预热，最大分别2048/1024；没有将67写作75，padding仍不超过scheduler缓冲。两组TopP16/Decode32真实路由及六份安装源相同；每个response usage4096/1024/5120及逐轮seed配对核验，CLI除预算值外相同。7项greedy相同但不代替新随机采样Math。

本地harness扩展1024参数/setup检查及预算对应预热数量；七个请求/计时/清理helper AST与原版完全一致，未改原小测或官方benchmark源。迁移hook路径的两次无server/无round失败先前setup_failures_32g保存，成功数据来自独立retry2，不覆盖失败或复用旧控制。

原始包4152244bytes/16分片，SHA256 eeb80b02e45c873b15d467182b875a2e38f3169d0dda282f9e0ebc050fa858bc；分片/整包、6份冻结源码、384usage/seed、实际75/67预热、CLI、mean及helperAST均本地核验。原件service-results/20261014_tianshu/budget1024_service/20261014_budget1024_service_retry2_tianshu。

结论：当前配置保留全局预算2048。负例不证明所有调度优化无收益；下一步可考察真实V1使用的单请求prefill阈值，在保持全局2048的前提下调整长请求公平性，先核对实际代码。现正式最好3506.84/TTFT4494.12/Math103/tag tianshu-best-topp-w16-20261014不变。性能剖析TODO图内kernel覆盖，安装源/模型权重/原benchmark不变，沐曦只读。
