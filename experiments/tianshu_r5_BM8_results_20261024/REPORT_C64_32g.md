# r5+BM8新C64：103/105，98.0952%，完整通过

新API原13项serve argv，1023来源/67显式环境，完整105题Level3；temperature1/top_p0.95/max_tokens32768，全部题和触顶题计入分母。控制器正常完成、独立collect_quality_cpu再次核验通过，103/105=98.095238%，请求错误0；逐题分数与实际stop/cap数量见C64_KEY_SCORES_32g.csv和C64_QUALITY_GATE_32g.json。原始预测/报告/日志/来源快照已完整压缩持久化到容器，SHA与tar比较见C64_RAW_ARCHIVE_32g.txt。

本次不是只有启动capture：全部42层BM8最终源码5cc055、网格[64,2,4]、BM8/Q1/T16/S4、严格非debug机器码匹配、167寄存器/0spill；完整profile_audit还把真实C64请求Graph重放与capture绑定，达到>=100次的预定门槛。实际最终计数和frame上下文留在容器原始profile_audit，不把Python调用数当作Graph重放。原r5 997来源、84 fitted权重bank、原生显存计数、Down/LM/GU/KV/采样全部保留并重核。

C64质量测试按原r5协议把未来KV准入切为control，以实际达到64路，不能把这个设置用于正式性能；4K/16K服务仍使用future candidate。新C8为100/105=95.2381%，stop103/max_tokens2全部计入、错误0，两个独立完整质量门槛均通过。

接下来进行同场新鲜输入4K N128四轮两组对照。若满足预定吞吐>=1%、热轮各<1%波动、TTFT退步<=2%，再原始N256正式4K。按用户新增要求，当前4K结束后最终选定版本还要运行一次官方16K [16384,1024,64,128]。当前尚无新吞吐，正式r5 6123.31及旧tag保持不动。
