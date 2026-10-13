# 天数 BI-V150 32g：整数位宽审计与实验总表

截至 2026-10-13 13:41，正式最好：4K 3222.52 total tok/s、644.51 output tok/s、TTFT4510.23ms；16K 2440.62 total tok/s、143.57 output tok/s、TTFT200982.18ms。独立 Math Level3 102/105、97.142857%、0 errors。新增实验均独立归档，未把微测成绩当正式成绩。

| 位置 | 改动与校验 | 算子证据 | 服务证据 | 当前处理 |
|---|---|---|---|---|
| FlagGems radix histogram 局部累加 | int64→32/16/8；80 histogram、120 stable sort/graph检查全部通过；全局计数仍32、公开索引仍64 | M64真实vocab随机asc：64→32完整sort约+6.115%；16/8无额外优势，物理寄存器同32 | 4K/1K C64 N64各5轮，比较末3：3253.05→3245.94，−0.219% | 不采用 |
| radix内部索引pingpong | 私有索引32，最终转64；120 CPU稳定排序/GPU/graph检查通过 | M64随机asc sort 24.954→21.398ms，约+16.62% | 同服务条件：3247.84 vs3253.05，−0.160% | 不采用 |
| radix私有prefix sum | scan32；公开API/浮点数据不变，数值及graph校验通过 | 几乎无速度收益 | 未单独追加服务测试 | 不采用 |
| 原生top-p | 审计实际采样入口 | 已用int32/uint32 | Tianshu batch≥8走原生Triton；只batch<8经过上述FlagGems排序。MetaX强制PyTorch路线不同 | 保持当前原生路由，不强制切换制造微测收益 |
| Decode attention元素偏移 | 28 stride annotations、6 offset casts缩32；保留64物理指针；所有张量带tile余量的offset<2^31；只B4..48/BF16/NONE | 实际交错KV+Qstride与expanded K/V scales的10场景全部FP32参考及与原实现逐位一致；B30/17K约+14.0% | 首次hook要求unused descales为空，全部回退，已停止且完整归档；修正后有真实B4/8/16/24/32/40/48路由。16K/1K C64 N64三轮中，r1=2252.28排除，r2=2563.69 vs控制r2=2375.55（单轮+7.92%），r3进行中 | 还不能称完整服务结论；候选尚未采用/新Math/新正式 |
| KV cache写入 | 已审计真实入口：stride和slot载入后使用64；准备32及w16/s10、w4/s1、w1/s1分开比较 | 尚未GPU运行，仅py_compile | 未运行；公开slot_mapping仍64 | 下一轮候选，等GPU串行空闲 |

模型输入req_indices、positions、query_pos还有64使用，但存在通用索引API、RoPE、slot mapping合约，不能全局替换。当前token/block metadata与原生采样大量已32。slot总数约52万、元素偏移约数亿，不能用16/8表示；signed32的安全界限必须逐张量确认，超界回退原64。

独立框架发现：冷首轮CPU栈主要在FlagGems配置读取的SQLAlchemy全库反射。自己的算子仓库已在本地Python3.13、真实只读902表快照核验只反射目标表，配置等价，现有表约4–5ms vs全库1.5–1.99s，缺失表直接返回约1.2ms。这里只是本地CPU证据，尚无天数服务收益；未混入attention对照，未更改缓存配置值或pool key。

证据路径：hist服务 `tianshu_sort_counts_20261013/results/20261013/service_32g`；index服务 `tianshu_sort_buffers_20261013/results/20261013/service_32g`；offset真实参数 `tianshu_attention_offsets_20261013/results/20261013/bf16_descales_32g`；SQL在自己的ops仓库 `tianshu_sql_reflection_20261013/results/20261013/local_cpu_v2`。所有沐曦仓库只读参考。原benchmark SHA保持0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565。

13:47更新：地址32服务三轮全部完成，计分 2393.194947→2562.170850，+7.060683%，候选计分轮波动0.118906%。30种实际路由、384吞吐成功、7greedy相同，全部raw已本地SHA核验。见service_32g报告；新Math和正式成绩仍待完成。
