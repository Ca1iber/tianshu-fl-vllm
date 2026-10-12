# BI-V150 32G：2步draft greedy、主模型随机采样实际验证

此前2步概率profile小规模1943.90 total tok/s，对同规格控制2731.42无收益。新候选使用安装版vLLM原生greedy draft选项：草稿每步argmax，q按one-hot处理；主模型仍temperature1/top_p0.95，标准随机拒绝与恢复采样，正FP32拒绝阈值。保留7槽骨干/2步前缀、KV预留7和足够缓冲。

首次smoke两请求成功，但原diagnostic key只有(batch,是否dense概率)，把目标greedy和目标stochastic且draft greedy两情况合并，导致等待器要求actual stochastic记录的检查失败。该问题只涉及记录去重，不是采样或返回失败；旧数据完整保留。改key与文件名包含目标all_greedy状态后，第二次两请求也成功，记录分别显示target_sampling_is_stochastic False/True，draft均one_hot_argmax，概率缓存为空，标准拒绝kernel保留。

v1/v2原始日志、响应、源快照均已分块及整包SHA256备份。v1包2065117 bytes SHA168f4b2148dbb73f6136100f7e564a20f10dd6bd5af7b415a61145b88e806d51；v2包2064284 bytes SHA6acba1189705192ff7d1e33f228f6d450b04dd9f8710f9ecac2f37af4a57a14e。

成功smoke后已自动进入C64/N128、4K/1K同种子三轮小规模对照；当前尚无完整吞吐/准确率。质量审计支持原生greedy或probabilistic draft，要求真实主模型随机路由、标准拒绝，仍严格核对官方temperature/top_p和105题无错误。不能改主模型为greedy来换分。

正式最好仍2712.55 total tok/s、TTFT4720.38ms、Math102/105，此候选未采用。性能剖析：TODO。官方benchmark、权重、安装源码和沐曦仓库未修改，未跑优化版完整16K。
