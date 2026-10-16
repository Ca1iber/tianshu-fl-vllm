# 新一轮固定gather真实API复验

不更改冻结f48351f旧实现，只在当前实际安装版本重新验证。48组FlagGems公开API涵盖1D/2D、BF16/FP32/int32/int64、int32/int64索引、连续/strided与负的有效索引；另3次改变索引/数据的Graph replay全部与CPU逐位一致。六组use_gems内真实Torch `source[index]`，覆盖宽1/2048/130560与int32/BF16，观测到固定路由且CPU逐位一致。4份源码与所有记录已下载核验；原始GPU大缓冲未逐个序列化。

该kernel没有autotune，row count/stride不做JIT特化。合法axis0单个1D整数索引外继续原路。此处只有功能证据，没有吞吐/Math结论；后续独立fresh ngram服务保持S8与其他参数，只增加此gather。归档同RAW_MANIFEST。
