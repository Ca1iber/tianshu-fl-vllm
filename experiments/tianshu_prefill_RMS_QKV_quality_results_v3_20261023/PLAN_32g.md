# 新prefill融合模型的独立CPU质量复核

全部读取本次C64/C8的105题report、review和prediction，保留错误与截断；严格>95%。独立重新执行完整原r3及新融合的所有启动/运行/数学校验，完整profile和accuracy_audit须逐字段相等。逐一复核模型/控制器当前文件与冻结快照SHA、原checkpoint SHA、serve参数/环境、GPU退出空闲。C64实际42层融合运行必须全部覆盖；C8不虚构未命中的M4096路径。不启动GPU、不覆盖任何原始记录。首次GPU前冻结该复核器。
