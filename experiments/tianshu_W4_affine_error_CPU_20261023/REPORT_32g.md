# W4非对称minmax量化的CPU误差研究

对真实Down第0/20/41层、3个完整GateUp fixture、group32/64/128/256，完整FP64投影共4718592格。参数先存为实际BF16：offset=min(weight)，scale=(max-min)/15，再按实际offset/scale量化到uint4 0..15；参考为BF16_round(q*S+Z)后完整FP64点积。对应已验证dotsaddz的数学语义，但新实际参数分布/新group尚未执行GPU，不能据此宣称native正确或有性能收益。

平均投影相对L2：G32非对称7.6959%，同G对称9.7938%；G64为8.9932% vs11.1529%；G128为10.2766% vs12.5256%；G256为11.5645% vs13.9692%。G32非对称相对当前G128对称的该3层3fixture平均误差约低38.6%，不是准确率提升38.6%。G32非对称最大误差8.2376%，仍非零误差，需要实际GPU完整数值检查和新鲜模型质量。

该CPU研究不使用torch/triton/vllm/flag_gems，不执行GPU/模型，不更改checkpoint或当前运行的G128候选，不输出大数组。控制是前一CPU对称研究同层同fixture的FP64误差表，其SHA写入门禁。它为若G128模型质量不足时提供独立候选；当前服务保持冻结998971f不动。所有代码/关键CSV/门禁/报告推天数GitHub。
