# 天数Decode组批元数据融合微测

原始ModelRunner方法与可逆生成方法在真实CpuGpuBuffer和GPU上对照，batch为1/7/8/16/24/32/40/48/56/64、两组种子，共20组；随后改变context及上一轮采样token再次对照。GPU全部字段及padding、CPU bookkeeping、logits_indices dtype与slot mapping入参检查均通过。positions和req_indices保留int64。六类不支持场景拒绝快路径。两次Graph重放改变live输入并先清空输出，通过非陈旧结果检查。

交错7组、每组20次，完整准备方法host wall中位数：原始2.0229667425 ms，融合0.3210245166 ms；GPU stream event分别1.6273059845与0.0053592309 ms。计时包括真实CPU/H2D，slot table仅是观察入参的stub，未运行真实slot mapping或模型，不能作为端到端增益。

下载后独立校验5份源码SHA、可逆生成结果、20组检查记录、6个guard、2个Graph记录、14组原始计时及中位数。归档未保存完整张量值，本地只能核对记录和源码，不能独立重新计算GPU输出。下一步真实服务3+3对照，原serve命令、KV布局及采样不变。

原始目录：/workspace/benchmark-results/20261016_decode_batch_meta_micro_tianshu。归档SHA见ARCHIVE_MANIFEST；复核命令：`python3 experiments/tianshu_decode_batch_meta_20261016/verify_micro.py --root <下载目录>`。正式最佳4093.81 tok/s、Math102/105保持。
