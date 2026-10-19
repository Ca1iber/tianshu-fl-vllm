# 保留原失败门槛的逐层定位

原混合图模型首测的3个N64/72query容量样本通过；N63/Q78→N64/Q80在完全unpadded eager比较中失败。same-padded eager hidden/logits逐位一致，KV位置/slot、42层守卫页、候选页恢复和RNG均过；unpadded hidden maxabs2.0/43181失败元素，logits maxabs.4140625/2574902失败元素。原件failed状态与95源码保持不变，不能发NumericGate。

本版本复用原冻结hook与numeric_probe、相同预设8样本门槛；只对自有N64/T64,72,80,88图添加GPU层输出clone，并在参考eager保存相同层输入/输出。42decoder输出、首层norm/qkv/rope/attn/o_proj/gateup/act/down输入输出、embedding/finalnorm纳入诊断。输入用forward_pre_hook保存以避开in-place修改。失败前保存图/same-pad/unpad的逐层标量差异，不放宽原断言，不当吞吐。clone与保持张量引用可能改变执行/内存条件，即使带观察器通过也不能给未观测原版本背书。

run_layer_trace沿用4096/512 C64N64两阶段与seed，增加逐请求完成正文JSON，确保某请求失败时其他已完成响应仍保存；原失败轮没有保存的响应不重构。单GPU、23:58硬停，尚未GPU运行。旧95源/累计136来源不可修改。
