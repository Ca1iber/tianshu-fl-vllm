# K4/V16：独立GPU组件，不修改已发布r5

数值基准为已冻结 tianshu_K4V16_contract_20261024/CPU_GATE_32g.json：每token/head 128维，RN32(maxabs/7)、RN32倒数、RNE取整、q范围[-7,7]、q+8双nibble，微小整行scale=1。V BF16位模式全部保留。

当前对照必须为 r5 f1df paired K8/V16。K4从该源码精确可逆变换，只修改3D主kernel的K解包、K逐token/head缩放和去除V scale分支；wrapper/reducer/softmax不改。使用原配置T16/W1/stage1与原segment政策，C33为S16、C64为S4。真实writer+reader+reducer同Graph计时，两组交替9轮各20次回放。额外分别测writer与reader以定位成本，但决策以完整组件为准。

先GPU writer primitive：两次穷举所有65280个有限BF16 K位模式、RNE半整数、符号零/微小值/边界；V覆盖全部65536 BF16位模式。实际原始页面整块bitwise比较、strided与guard页、输入padding、负slot、同Graph原地变数据和RNG保护。不得修改CPU契约去掩盖失败。

随后真实分页decode：4K/5120，C8/C33/C64，ragged/strided，以及真实16K驻留批次C33长17408；完整GPU FP32量化参考，并用保存的完整量化输入在CPU FP64重算所有输出，另报相对未量化K的近似误差。全部输出、页面、slot、scratch、RNG校验；原始Q/K/V/长度/页表改变后复用同一个Graph，禁止只验证初始化状态。

实际raw页10496B，K2048B、FP32 scale128B、padding128B、V8192B。独立typed views必须共用单一存储。原生FullAttentionSpec计数只证明Spec预算可表达，不代表引擎allocator已接入或模型已释放显存。已发布r5仍保持原12288B页。只有组件正确且常用4K/5K/strided有一致收益才接模型，重新全量Math>95并正式4K。不再例行重复完整16K。本轮最晚2026-10-24 23:58北京时间停止GPU。
