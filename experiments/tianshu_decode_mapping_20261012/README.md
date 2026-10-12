# 天数纯 Decode 查询块映射实验

源码事实：native unified attention 已按KV head组共享8个Query head，不能把GQA共享当作新优化。现有查询块网格采用上界 q.shape[0]//BLOCK_Q+num_seqs，并通过resolve_seq_and_query_len查找序列。MiniCPM的BLOCK_Q=2、每序列最多1个Decode query时，上界通常比实际序列数多约50%，含空块。

候选仅改变纯3D Decode的块映射：pid0直接指向序列、局部查询块索引0；网格第0维变为num_seqs。完整保留原生Q/K/V算术、mask、softmax、分段写入和reduce，不可仅截短原始网格（原始映射有序列间空洞，会遗漏后面的序列）。

实现先核验原生源SHA，用明确匹配的两处修改生成独立源码副本，Triton从该真实文件读取函数体。安装源码保持原样。数值实验对比当前adaptive分段（B1S64、B2-15S16、B16-64S4），包含随机分页、非连续缓存/Query、ragged KV、0/1查询长度和输出/scratch尾部保护；全量FP32与选定CPU对照、逐位native对照，交替CUDA Graph event计时。

该实验尚未跑，不能声称已有性能或质量收益。数值/实际服务/对应Math105通过前不进入正式最佳。串行排在扩展FA2筛选之后，受预约截止约束，不运行完整16K。
