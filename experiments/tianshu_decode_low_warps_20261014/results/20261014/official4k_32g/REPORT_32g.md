# 天数 BI-V150 32g：Decode两warp正式4K达到3640.16 tok/s

原始官方脚本、4K输入/1K输出/C64/N256，共4轮并只排除首轮。正式均值3640.16 total tok/s、728.03 output tok/s、TTFT 4465.17ms；较此前3506.84提升3.801713%，较公开2028.01提升79.494184%。新的同源独立Math103/105=98.095238%、API errors0、105自然stop，严格>95%。保存为当前正式4K最好记录。

|轮次|total tok/s|output tok/s|TTFT ms|duration s|计分|
|---:|---:|---:|---:|---:|---|
|1|2227.34|445.47|7032.72|588.47|否|
|2|3495.1|699.02|4462.02|375.02|是|
|3|3715.77|743.15|4464.29|352.74|是|
|4|3709.6|741.92|4469.2|353.33|是|

1024请求成功来自原脚本四轮CSV汇总，逐轮256、input1048576/output262144严格匹配；本次核验范围为原始聚合计数和token totals，不声明逐请求正文验证。后三轮全部保留，极差/均值6.062102%，波动尚未达到0.1%；第2轮3495.10较低也没有剔除。三轮小测+4.046324%、独立五轮+3.999652%，本次正式+3.801713%方向一致，但先前正式最好与本次不是同期配对正式控制。

改动仅Decode batch8..64从warp4改warp2，KV32/stage1无sink L0及生产/合并tile32配对保持；其余batch保留warp4。基础TopP16/nativePrefill M256/KV128/w16/s1 L0/B75G1/budget2048、BF16及64位地址保留，无量化、投机、IRRMS或RoPE候选混入。真实B64 compiled warp2、57组TopP16、完整producer/reducer配对和75点预热/fixed int32 gather核验。

原benchmark SHA0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565不变。四份运行源snapshot、四份Math源/六份安装源、实际Math与正式server argv、原CSV/均值、实际kernel路由/75warm均在下载后独立审计通过。source CD3正式runner/audit4a09、实际主hook6aee/Decode hook4d5c与新Math完全对应。launcher bash语法和截获实际argv/14计算开关与本次两个真实argv匹配；新launcher位于本实验目录，旧best启动文件保留。

正式原始包2729526bytes/11分片，SHA256 8151569c414d834d58f199042e7c100d5352fdfd639fb4c8da8f68cba7a61181，所有分片/整包已核验。新Math包3435235bytes/14分片，SHA256 be54fa156540bdff9857b0ce2bd119338b6bf33cd4162f5f99f0d97fc71cdf9e，105原始预测与评审已核验。原件service-results/20261014_tianshu/decode_low_warps_official4k及decode_low_warps_math。报告CSV采用LF，归档内原始CSV字节保留。

结论：采用此完整质量/正式通过的4K记录，继续优化。该profile没有新16K，不将旧address32/stage2的16K分数并入。硬件剖析：TODO有效图内kernel覆盖，未据共享内存元数据推断硬件瓶颈；安装源/权重/benchmark不变，沐曦只读。下一项先测本profile的Decode分段数量，CPU/GPU参考、尾部/Graph检查通过后才考虑服务实验。
