# Ngram 动态 gather 稳定性

原生 ngram 初次4K/C64/N64轮完整成功，但1148.49 total tok/s、TTFT59.46s，并出现334次在线调优。首轮原本排除，后续已停止用于诊断，不能报稳态/正式成绩。没有以高接受率推断吞吐，也不把现场调优扣掉后冒充实测。

重新验证既有固定 gather：真实FlagGems public index API的48组（dtype/stride/int32/int64/负的有效索引）与3次变更输入Graph，加真实Torch API在use_gems内的六组路由与CPU逐位检查。通过后才在ngram模型加载后、profile/Graph之前安装。只接管合法 axis0 单个一维整数索引；其他形式保留原实现。不更改 attention、采样参数/RNG、接受比率或构造假缓存。
