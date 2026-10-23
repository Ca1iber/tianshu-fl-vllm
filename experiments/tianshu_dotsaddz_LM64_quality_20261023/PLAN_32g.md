# 新候选：Down64 W4 + LM64 W4，先全量质量

模型934来源保留916单Down父版本；新数值变化仅LM64。M8/其他M和GU委托原backend。C8沿用全部BF16 LM8审计，不冒充已覆盖LM64；C64显式以真实新W4 runtime>=10次替换旧W8计数要求，原43W8权重签名/全部GU路径、框架/KV/sampling/attention/Down新旧审计均保留，不伪造或修改父记录。保留真实native admission control的C64环境，只与C8差futureKV模式。

105题Level3温度1/top_p.95/max_tokens32768，先C64再C8，各自严格>95%，所有cap/errors完整105分母；任何失败不原样重跑追分。先CPU准备与SHA冻结，待单Down正式评测结束和GPU空闲才启动。截止UTC2026-10-23T15:58（北京时间23:58）。此前所有质量不能替代本版新质量。质量合格后另行短4K，未合格不得跑吞吐。不跑16K。

现有packed LM bank在容器，不新建矩阵数组；新增两轮原始文件预计<350MiB，执行前要求至少1GiB可用空间。全部原始数据/模型/缓存只在容器，GitHub仅代码/关键结果/报告。没有修改单Down正式测试的冻结文件或MuXi仓库。
