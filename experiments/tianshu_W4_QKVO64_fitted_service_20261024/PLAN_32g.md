# QKV/O64 fitted W4：联合模型验证协议

仅原 r4 的 FULL uniform64 decode QKV/O 投影改为独立组件通过的拟合 G128 W4A16；prefill、其他批次、采样/RNG、GU融合、Down和LM量化、K8/VBF16、前缀缓存和调度均继承。Checkpoint 不改，拟合仅使用原权重，不用激活、题目、标签或微调。84 组原始 BF16/既有W8保留，同增210,567,168字节派生缓冲；控制组装载相同缓冲但decode走原BF16。不得声称净显存节省。

O组件Graph42层1.23656→0.84544ms，QKV1.46686→1.15529ms；两项完整GPU数值及拟合误差证明独立完成。这些不是联合模型的正确率或吞吐，不借用O minmax的102/105。先做CPU来源/84银行计划/保护范围和内存计入核验，未通过不得启动模型。

原13参数启动、max_model_len131072、max_tokens32768、temperature1/top_p.95、完整105题，全新C64和C8分开启动API，全部题与max_tokens均计分。C64仅将未来KV准入设control以确保真实满批64覆盖；其他配置一致，真实图重放至少10次。任一质量≤95%不推进性能。

内存：运行时从原权重确定性重建84组拟合W4，全部原始/packed/scale/zero字节SHA须匹配组件；在实际DeviceMemoryProfiler装载窗口分配，实际peak profile内覆盖84native调用，CPU/GPU RNG无变化。记录全部模型唯一storage、真实KV块数，不指定手工KV容量。

质量全部通过后只做4K筛选：N128/C64跨轮新seed、配对对应seed相同，每轮真实前缀命中≤1%、零抢占、4轮排首轮，双方热3轮极差/均值<1%且配对提升≥1%才做原脚本N256完整4K。原benchmark不改，默认不跑16K；报告记录所有轮次及失败。2026-10-24 23:58北京时间硬停止GPU；性能计时期间不做拟合准备、哈希、压缩或push。
