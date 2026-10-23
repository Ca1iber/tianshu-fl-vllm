# dotsaddz 非恒定矩阵格式探针

原生 generic INT4 GPU 日志建议改用 dotsaddz 获取速度。本实验只检查该实际接口的物理格式，不改变checkpoint或服务，不测模型质量，不声称速度。使用前版已完整验证的 M64 K256 N256 非恒定2组logical矩阵，完整INT64 CPU参考（16384输出）。

事前固定30个假设：沿N打包的natural/permuted两种；沿K打包的KN/NK各natural/permuted四种，共6布局。每种枚举5种语义：unsigned q + additive -zero*scale；unsigned q以8为中心 + additive(8-zero)*scale；signed补码(q-8)&15 + additive(8-zero)*scale；unsigned q + 原始zero；signed补码 + zero-8。scales和zeros均BF16 [2,256]；N显式固定256。每种物理布局先在CPU完整逆变换检查。所有候选都必须和相同完整INT64参考逐格BF16精确匹配，不允许事后更换参考或误差阈值。

每个假设独立GPU进程，45秒超时，输入数据带初始化保护空间并校验全字节和版本，不复用损坏上下文。操作完成才能存全量16384输出并独立CPU复算。全部错误的假设仍保留证据；没有匹配只说明这30个假设未匹配，不宣布硬件不支持。GPU各进程串行，先确认卡空闲；最迟北京时间23:58结束。所有代码、输入与厂商库SHA在首个GPU前冻结并提交。无大数组上传GitHub。
