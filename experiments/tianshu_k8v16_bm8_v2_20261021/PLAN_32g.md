# K8V16 decode：BM8与直接序列映射分因子验证

当前f1df reader真实C64是T16/W1/stage1/S4，BLOCK_M16、每KV8query heads、BLOCK_Q2，grid96×2×4。安装后端min_dot_size允许小M/N(以硬件填充)，BF16 dot收缩K仍需16，本实验保留T16，不是此前失败的大tile或BM8 SIMT实现。
四臂current/BM8/BM16_direct/BM8_direct；BM8仅num_seqs>24且3D/GQA8/D128，BLOCK_Q1并真实修正grid。direct仅3D(原wrapper强制max_query_len1)，将seq=program_id，读取原相同qstart/长度，grid精确num_seqs；2D原解析器保持。原dot/softmax/量化codec/写入/reducer文本全部保持，独立比较映射交互，不能预称BM8硬件计算减半。
沿用已核验真实KV writer、14场景×两次动态同Graph回放、full GPU FP32/CPU选中GQA向量、真实storage和scratch guards/RNG。实际记录BM/BQ/grid/regs/spill/二进制；112输出Tensor及304条CPU向量独立复核，冻结前不运行GPU。
先等当前471版阶段诊断退出并确认GPU空闲；不得同时占卡。稳定4K组件显著提升才接整模型；所有代码/报告/关键数据放容器和GitHub，23:58北京GPU硬停。

V1首次四组n1/n8已通过，进入n32时被原组件启动器固定BM16/BQ2断言拒绝，尚未编译新BM8。V2独立派生native_launch，仅将这个断言扩到精确的(16,2)/(8,1)两种实际ABI；Producer其余AST与Reducer完整AST分别对原件归一化验证不变。V1源码及结果保持。本次不是放宽数值误差/guards或将失败改为通过。
