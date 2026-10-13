# 天数 BI-V150 32g：native top-p启动参数及CPU参考

实际MiniCPM词表130560，原native top-p源SHA64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8；仅比较warps/stages，kernel body、block8192/trunc4096、grid、温度/top-p/top-k和RNG不改。计时包含相同输入恢复copy，轮转Graph七次，每次十次回放。7个输入覆盖B8/32/64、normal/peaked/部分-inf/ties，以及top-k100+top-p0.95。

|输入|原w4/stage3 ms|w16/stage1 ms|同原版过滤结果|
|---|---:|---:|---|
|B8 normal|1.609024|1.064357|逐位相同|
|B32 normal|3.534040|2.348392|逐位相同|
|B64 normal|6.985896|4.552956|逐位相同|
|B64 peaked|3.923025|2.853634|逐位相同|
|B8 masked|1.758705|1.152341|逐位相同|
|B8 ties|1.162725|初版未计时|一个过滤位不同|
|B32 top-k+p|0.842897|0.555773|逐位相同|

初版所有场景逐位一致的方案只有default及w4/stage1，后者无明显性能收益。w16/stage1普通B64约快35%，但tied logits有一位不同，不能宣称所有输入bitwise等价。w2部分场景有13位差异，且明显更慢。

第二份独立探针补CPU FP64参考：先从GPU传CPU，再CPU double转换，避免已观察的CoreX GPU FP64转换异常。检查概率质量达到p、移除最小保留概率后不超过p、排序边界和至少一项保留，严格误差1e-6，容差未放宽。

严格CPU结果没有任何方案在所有7场景通过，包括原版default。normal/peaked的原版和w16候选掩码相同，其CPU质量误差也相同，最大观察短缺约0.000588；不能把此结果写成CPU高精度全部通过，未定位该原版偏差的底层原因。唯一w16改变的ties场景，独立CPU概率质量/最小集合/排序边界检查通过；与稳定CPU精确排序仍有一位差异，但在1e-6范围内满足门槛。

服务试验采用“输出与原版逐位一致，或改变的输入通过独立严格CPU语义检查”的相对原版保真门槛，不修改1e-6，不隐藏原版严格误差。w16在7项符合此门槛；w2的B64 normal变化未通过，排除。新的独立Math仍为采用前必要门槛，不能复用当前DecodeL0KV32的103/105。

真实vLLM0.24 bench不再强制temperature0；未传采样覆盖时，实际模型generation_config及已运行正式server日志均确认temperature1/top_p0.95，所以该路径与正式负载相关。下一项是在已验证DecodeL0KV32/B75G1/budget2048上做4K/1K C64/N64三轮排首的同期服务A/B。当前没有新服务或正式分数，不采用；正式最好3426.62保持。

初版2234699bytes/SHA32d24266f5ea675d8830749e5fcaa3533c2cf61dfa475be4b27e58105fc6a574；CPU参考版2236727bytes/SHA3c73d344fed37bfc6770cf60a3194f20922130f91b01b649ad5da724fe1a9e34，分片/整包均本地核验。CPU版源码与提交源逐字匹配，初版旧源独立保留于原始source_snapshot，未追溯替换。完整原件在service-results/20261013_tianshu/topp_launch_{initial_micro,cpu_reference}。硬件性能剖析未采集，TODO。安装源、权重、官方benchmark不改，沐曦只读。
