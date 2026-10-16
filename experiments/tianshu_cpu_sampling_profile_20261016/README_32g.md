# 天数vLLM CPU调用诊断

准备中，须在无其他GPU工作时独立启动。原始官方argv与当前最佳原生TopP/CV/KV16/S8；4K/256输出C64，先暖一组，第二组预暖32个纯Decode后采16个FULL/C64 frame。分别cProfile execute_model/sample_tokens，记录完整父调用、host wall/thread CPU和源码快照，无GPU事件。诊断有侵入性，不能当吞吐或GPU活跃时间；父子累计时间不相加。
