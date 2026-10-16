# 天数vLLM CPU调用诊断

准备中，须在无其他GPU工作时独立启动。原始官方argv与当前最佳原生TopP/CV/KV16/S8；4K/256输出C64，先暖一组，第二组预暖32个纯Decode后采16个FULL/C64 frame。分别cProfile execute_model/sample_tokens，记录完整父调用、host wall/thread CPU和源码快照，无GPU事件。诊断有侵入性，不能当吞吐或GPU活跃时间；父子累计时间不相加。

同时只读记录16帧的logits、完整hidden states、sample hidden states与temperature/top_p/top_k的形状、stride、数据及storage指针，以及随机采样/penalty/logprobs等元数据。没有读取GPU张量值或额外同步；用于核对是否适合把logits投影和采样一起捕获进CUDA Graph，尚无实际指针稳定性或收益结论。实际SamplingMetadata源码另入冻结快照。该诊断尚未运行，不影响当前Math的58+10份冻结源码。
