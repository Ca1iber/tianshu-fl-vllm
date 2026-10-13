# 天数 BI-V150 32g：真实native RoPE融合服务仅提升0.31%

真实模型命中75组Q/K融合kernel，40项native/CPU逐位数值与Graph/alias检查通过后，同期4K输入/1K输出、C64/N64各3轮，预排首轮。控制3514.761476→候选3525.631179 total tok/s，提升0.309259%。收益小，暂不采用、不启动新Math或原正式benchmark。

|轮次|控制tok/s|候选tok/s|控制TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1253.344802|1313.334522|25801.862824|19021.068646|否|
|2|3521.028084|3523.642676|13297.560017|13383.573565|是|
|3|3508.494869|3527.619683|13690.950784|13431.490910|是|

384/384请求成功，usage4096/1024/5120及逐轮提示seed列表配对一致；7个greedy探针相同，但不替代随机采样质量。两组TopP16/Decode32/nativePrefill/B75G1/budget2048及六份安装源一致，不加入IRRMS候选。实际候选75条RoPE source04d记录/stride2560，控制0条，命中有效。

两组都用VLLM_DISABLE_COMPILE_CACHE=1和各自独立cache_dir，保留Inductor/FULL_DECODE_ONLY，实际CLI除缓存目录外完全一致。首次模型被旧AOT图绕过的失败另存；本次真实重编译和命中可证，不能把未命中的控制复用为实验候选。首轮候选TTFT明显更低，但预排首轮规则保持，不用该冷启动变化当稳态成绩。

全原始包4177436bytes/16分片，SHA256 1d4fe0ba78f568ab37061e251375f81fb4ccd2dc49abb32d3e8f2dd492f069c7；整包/分片、384响应、7冻结源、实际路由/CLI及均值均已本地核验。原件service-results/20261014_tianshu/native_rope_service/20261014_native_rope_service_tianshu。

结论：本版本端到端增益有限，不进行全量质量/性能评测，继续其他优化方向。40场景逐位匹配不能宣称所有形状或整模型全局相同；若未来改变实现再独立数值和Math105>95。当前正式最好3506.84/TTFT4494.12/Math103/tag tianshu-best-topp-w16-20261014保持。性能剖析：TODO图内kernel覆盖；没有新16K，安装源/权重/原benchmark不变，沐曦只读。
