# 天数 BI-V150 32g：B1/B16分段策略服务无吞吐收益

同期4K输入/1K输出、C64/N64各3轮并预排首轮，两组都是当前L0/KV32/warp2-B8..64（其余warp4）/TopP16/B75G1/budget2048基础。候选仅B1 S64→32、B16 S4→16，其他batch原策略不变。控制3664.478656→候选3662.629752 total tok/s，变化-0.050455%。无端到端收益，暂不采用，不追加新Math或原正式benchmark。

|轮次|控制tok/s|候选tok/s|控制TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1332.979749|1321.691583|19042.638987|18837.298727|否|
|2|3665.587579|3661.010988|13168.218505|13263.853723|是|
|3|3663.369733|3664.248517|13248.572439|13212.333200|是|

384/384请求成功，逐轮seed、实际usage4096/1024/5120和所有原始文本本地核验；7个greedy文本相同不替代官方随机质量。七份冻结源、六份安装源与实际server argv匹配，双组75点真实预热/TopP16及生产合并tile32配对通过。前置micro及本次服务不混入地址32、IRRMS/RoPE、量化或投机。六轮没有新增调优记录。

实际chooser记录B1由64→32、B16由4→16，GPUproducer/reducer均命中对应segments，B64维持S4/w2，B1w4/B16w2。新policy保留原adaptive configure的geometry、threshold、共享S64 scratch和回退guard，仅进程内包装choose_segments；COUNTS添加S32键避免遗漏。旧源码日志文字仍写原策略，不使用该静态文字代替实际新路由证据。

前置8长+14短+额外14 B16边界共36场景/108候选通过CPU选定参考、GPU FP32/native容差/尾部/Graph；B1和B16的局部算子收益没有转化为本负载整体吞吐。不由此断言所有分段策略无收益，不把局部15%写成服务提升。

全原始包4226819bytes/17分片，SHA256 4f6bc339338aa89350d31cf86ae0f88b7e9b2fa1894a78bf036459626c666f5a。整包/分片、384原始响应、七份源及实际chooser/GPUkernel/CLI/75warm与均值已本地核验。原件service-results/20261014_tianshu/decode_segments_service/20261014_decode_segments_service_tianshu。

当前正式最好3640.16/728.03/TTFT4465.17/newMath103/tag tianshu-best-decode-kv32-w2-20261014保持。下一项重新检查当前L0/KV32/warp2基线的有界32位地址，与旧KV16/warp4基线不同；先单算子正确性后ROI，不更改当下最佳。无新16K服务。硬件profile：TODO有效图内kernel覆盖；安装源/权重/原benchmark不变，沐曦只读。
