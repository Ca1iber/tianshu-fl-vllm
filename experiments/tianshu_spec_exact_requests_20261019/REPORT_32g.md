# 保持真实请求数的混合图：模型数值门槛通过

2026-10-19，BI-V150，源码提交 8b450b4。保留原骨干编译和数学算子，只增加248个 mixed Q1/Q2 图键。请求数 N=16..64 保持真实值，token 以8向上补齐，额外 token 不超过12.5%，其他情况保留原执行路径。

八个真实模型样本分别覆盖 N64 三个、N63 三个、N56 两个；真实 query/capacity 为70/72、70/72、71/72、84/88（三个）、68/72、72/72。各样本输入/位置哈希不同，包含同一图改变输入后的重放。所有真实 token 的完整 hidden 与 logits 对同补齐及完全未补齐 eager 都逐位一致，最大误差均0；无需放宽 .02/.01 门槛。位置、GPU block table/slot 和 query 边界校验通过，42层候选 KV 恢复、非目标页哨兵和 CUDA RNG 不变校验通过。

两轮4K/512、C64/N64共128请求和4个固定种子canary全部成功。实际 mixed 调用中799次走已捕获图，144次保持eager，覆盖率84.7296%；另外70次all-Q1、262次prefill、2次empty单列。这里是host replay分支覆盖，没有宣称GPU逐内核追踪。所有metrics无preemption。由于有额外reference forward与归约，本轮耗时不能作为性能成绩。

原始归档5,344,262字节，SHA-256：`5e063d8d83b4d75b579bf014b721fd7a926a7dcda1a84a1682e277c97685d26b`。本地已校验95份源码、42项实际环境、128条完整响应与4个canary及metrics，见service-results/20261019_tianshu/exact_model/20261019_exact_request_graph_model_probe_tianshu/local_download_audit.json。

据真实结果生成精确请求数服务实验的NUMERIC_GATE_32g.json。下一步为相同原生ngram栈的3+3轮4K/1K小规模服务对照，首轮排除；尚未运行新Math-500或正式benchmark，未采用，正式最佳4093.81 total tok/s/Math102/105保持。
