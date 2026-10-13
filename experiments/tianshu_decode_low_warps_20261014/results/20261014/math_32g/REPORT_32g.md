# 天数 BI-V150 32g：Decode两warp独立Math105正确率98.10%

独立重跑官方Math-500 Level3，103/105=98.095238%，严格>95%；105/105请求成功，API执行errors0，全部105自然stop，错题18/87。虽然分数与旧warp4版本相同，本次使用新workdir与新API独立生成/评审，不复用旧103/105。输入来自官方路径，batch8、temperature1、top_p0.95、max_tokens32768、timeout3600与官方参数一致。

候选仅batch8..64的Decode为KV32/w2/stage1 L0，其余batch继续warp4；TopP w16/stage1/nativePrefill L0/B75G1/budget2048保持。真实B8 compiled metadata=2warp/1stage、实际TopP16与75点预热/fixed int32 gather核验。生产与reduce_segments的batch/segments配对，合并tile32，generated源4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e，与五轮服务候选一致。无新地址int32、量化、投机、IRRMS或RoPE候选混入。

105唯一预测与105唯一评审、逐题成功状态/计分/stop、报告execution summary、四份实际源码snapshot、六份安装源、实际server argv与官方评测参数，以及源/路由/75warm均已在下载后独立核验。run_math_eval源82cc8196457940352f6b445a248d20df6a05b33b56262b4033224194ec1530cc，audit源4dee9fffdb680a10c25b01b2846b8dc0a9930d28242ed89ae6f96e5f0104ccd6；实际主hook6aee与Decode hook4d5c为已确认五轮版本。新增verify_downloaded_math.py执行实际原始包检查通过，未用合成样本模拟质量。

全原始包3435235bytes/14分片，SHA256 be54fa156540bdff9857b0ce2bd119338b6bf33cd4162f5f99f0d97fc71cdf9e。整包/全部分片及原始报告、预测、评审各SHA已本地核验。原件service-results/20261014_tianshu/decode_low_warps_math/20261014_decode_low_warps_math_tianshu；完整原始回答保留，不重构缺失正文。GPU/API/evaluator清理已核验。

五轮小测3526.692389→3667.747812 tok/s，+3.999652%，本次新质量通过后进入原正式4K N256四轮、排除首轮。正式结果尚未产生，当前正式最好3506.84/TTFT4494.12/tag tianshu-best-topp-w16-20261014保持；不把3667.75小测当正式分数。没有新16K。性能剖析：TODO有效图内kernel覆盖；所有安装源/权重/原benchmark未修改，沐曦只读。
