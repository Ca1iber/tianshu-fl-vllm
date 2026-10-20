# Prefill/混合批次 INT8：模型4K小测 +1.75%，进入新质量验证

2026-10-21，BI-V150 32G。两组原13项启动参数、407运行来源/62环境、4096预算/64序列/async。两组相同纯decode NN M8/M64；候选仅24个已数值预热、128对齐contiguous大M矩阵用A8/W8，包含混合批次的全部行；其他M/布局BF16，LM保持原路线。两组都预热同一24形状，GPU/CPU RNG与线程数保持。此对照基线为NN decode，不冒称直接对照正式默认版。

| 组 | 轮 | total tok/s | TTFT ms | 新调优 |
|---|---:|---:|---:|---:|
| control | 1，排除 | 5230.32 | 13029.91 | 0 |
| control | 2 | 5429.65 | 10877.13 | 0 |
| control | 3 | 5421.60 | 10916.15 | 0 |
| candidate | 1，排除 | 5296.00 | 12479.01 | 0 |
| candidate | 2 | 5524.51 | 10363.03 | 0 |
| candidate | 3 | 5516.77 | 10375.06 | 0 |

计分均值5425.622762→5520.640252 total tok/s，+1.751273%；TTFT10896.638383→10369.043518ms，-4.841813%。组内吞吐差0.148410%/0.140154%，提升大于本轮波动。384计时请求成功、usage完整、零抢占、FutureKV全释放。候选42层实际M4096新路径、M128/2048/4096实际INT8与源/配置/二进制验证通过。

两组43-bank payload均1,326,938,112字节；KV40649/40731 block，不能因payload相同便声称实际KV容量相同。当前C64/4K/1K无容量抢占，没有新内存profile因果归因。路由计数5586/42等取自最后持久化的power-of-two快照，是Python调用下界，不能称全程最终计数；prefill并非Graph重放，纯decode Python计数也不能当Graph重放数。

未计分种子canary独立cache_salt强制完整前缀重算，每组seed可复现；跨组因量化可变。所有计时请求保持原参数、不带seed/盐。声明这项变化是为了同计算路径的复现检查，不在策略内读取seed/盐/内容选路线。

尚未采用。已准备新105题Math（严格>95%）和原始N256×4轮正式4K，CPU预检查证明两者运行407来源、62环境、13参数及额外CPU controllers完全一致。先Math再正式；不复用旧准确率、不跑常规16K。小测5520.64不是正式新成绩，当前正式推荐5482.79及高吞吐备选5509.23保持。

SCREEN_GATE_32g.json / SCREEN_ROUNDS_32g.json / KEY_ROUNDS_32g.csv是关键结果；完整原件仅容器 /workspace/benchmark-results/20261021_prefill_int8_service_{control,candidate}_tianshu。
