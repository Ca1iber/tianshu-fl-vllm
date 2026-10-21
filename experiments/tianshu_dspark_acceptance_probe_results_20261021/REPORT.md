# DSpark 短输出接受率：自然文本明显高于随机输入

本轮是诊断，不是新吞吐或Math准确率成绩，不替换正式5586.28推荐版。旧C64/N128/输出1024的0.5944接受数也不与本轮C8/输出128混为同一负载。

使用8道真实Level3题的chat-template输入，长度为[55,20,28,42,33,56,21,36]；随机匹配组逐条长度相同；另8条4096随机token，沿旧小评测区间[1,119999]，不声称逐位等同官方数据生成器。每组C8、T1/top_p.95/ignore_eos，输出128。先各16输出暖身，再ABC/CBA各一轮，同轮跨组对应seed相同。共72响应完整保存并独立CPU复核，包括暖身；两轮测量48响应，每条输出128。

| 组 | 第一轮每验证接受数 | 第二轮 | 按验证次数合并 |
|---|---:|---:|---:|
| 自然数学题 | 1.711640 | 1.785326 | 1.747989 |
| 等长随机输入 | 0.201176 | 0.273183 | 0.236044 |
| 4K随机输入 | 0.162286 | 0.213604 | 0.187391 |

自然组746次验证接受1304个草稿token，位置0/1无条件接受率91.96%/82.84%；等长随机组1648次验证接受389个，19.78%/3.82%；4K随机1713次接受321个，15.47%/3.27%。两轮方向相同。样本只有8个固定数学prompt、两个seed轮次，不能推广所有自然/随机数据或用短输出代表正式1024输出。自然与随机接受率差异支持工作负载内容是重要因素，尚未证明所有模型计算语义等价，也不声称整个投机框架没有优化空间。

实际运行42目标+5草稿层、BF16共享LM head、辅助层[2,11,21,31,40]、七槽backbone/两步提案、probabilistic dense draft概率与standard rejection。请求阶段记录验证新的七槽长度hook真实生效；FP32阈值在(0,1]，原rejection kernel/ratio不改，无合成accept。此前160项GPU输入准备及312 CPU长度边界校验的policy用于本服务；本服务请求不在maxlen末端，不能声称服务端真实越界请求也已验证。

原始目录 `/workspace/benchmark-results/20261021_dspark_acceptance_probe_GPU_tianshu`。33项primary source清单（并非全Python传递闭包）、CPU gate、源快照、全部输入/响应/metrics均复核。独立CPU重新解析counter与全部usage/prompt SHA，核对两步提案/逐位置计数恒等关系；全部通过。结果、精确计数、实际运行记录见 [ACCEPTANCE_GATE.json](verified/ACCEPTANCE_GATE.json)，逐组表见 [KEY_ACCEPTANCE.csv](verified/KEY_ACCEPTANCE.csv)。源冻结后未更改。服务trap已清理，ixsmi恢复68MiB、无运行进程。

不根据这一轮贸然投入完整投机benchmark。后续保留已验证多查询attention和七槽policy供真实投机集成；同时继续检查推荐主线实际采样的GPU开销。原生C≥8已使用Qrita Triton top-p，而非全词表PyTorch排序，不能重复提出“去掉全词表sort”这个错误前提。可先测试原数值JIT不变、仅launch warp/block配置的组件收益，再决定是否值得服务A/B。先读旧采样融合/Philox/Graph报告，避免重复已有+0.3%或负收益实验。
