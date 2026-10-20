# W8A8 fresh MATH-500 Level 3

2026-10-20，原105题、C8、temperature1/top_p0.95/max_tokens32768，100/105=95.238095%，严格大于官方95%门槛。105请求均有完整prediction/review、无API错误；103个stop、2个max_tokens。总输出358306 token，最大32768，中位1440。没有复用旧103/105。

独立核验全部逐题分数、usage、原始行/文件SHA；Git只存分数、用量与哈希，完整回答留天数容器。实际C8 model Graph捕获并回放42个GateUp整数分支，Graph外LM head实际执行整数GEMM。43组原BF16/INT8/scale哈希与成功screen及正式4K一致；layer0 GateUp和LM头也与真实权重组件一致。原BF16 checkpoint与embedding未改，新增非持久权重1326938112bytes已计入真实FL模型memory profiling。

同311份来源、9个控制器、57项实际环境和原13项argv的完整4K已完成：5375.50 total tok/s、TTFT4191.14ms、1024/1024成功、零抢占。质量与正式阶段的实际环境、源码及全部43组权重均独立核验一致，正式采用证据见tianshu_dense_w8a8_official_results_20261020/OFFICIAL_GATE_32g.json。此前5235.52/103题质量属于前一profile；新profile完整16K仍未测试。

原始目录/workspace/benchmark-results/20261020_dense_w8a8_math_tianshu；完整4K目录/workspace/benchmark-results/20261020_dense_w8a8_official4k_tianshu。
