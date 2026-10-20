# KV容量组合4K：5033.55→5036.15，+0.052%，质量待验证

原生max_num_seqs默认256→64，与预填充暂存区512MiB→128MiB组合；预算4096、K8/V BF16、S16策略、采样器和原始serve命令保持一致。两组3轮4K/1K C64 N64，预先排除首轮。控制三轮总吞吐为4743.36（排除）、5031.21、5035.88；候选4750.38（排除）、5040.98、5031.33。稳态均值5033.55→5036.15 total tok/s，+0.05177%，基本持平；平均TTFT11.5928→11.5896秒。

每组192请求、合计384/384成功；两组251份冻结源码完全相同、50项归一化环境仅两个方案开关不同。真实原生调度上限、Graph捕获与重放、暂存区在显存profiling内预留均经运行记录核验。实际KV块42386→44007，即678176→704112 token容量；4K均无抢占。

原始pair汇总在跨配置seed=777输出文本完全相等检查失败：每组内部两次重复均一致，但两组之间不同。原始state.json的failed/AssertionError、日志、逐请求文本及251份输入源码保持原样。这里使用新增独立分析器，不重跑或改写原始实验。分析器将跨配置文本差异明确记录为诊断项，并保留输入/API seed、用量和输出哈希。相同源码不足以证明跨配置计算逐位相同，差异原因尚未确证。

固定seed文本差异不等价于Math准确率；本筛选仍为quality_pending=true、fresh_Math_not_run=true、adopted=false。若后续16K有足够收益，必须跑新Math并严格>95%，再做同配置完整官方评测。4K数据不能当作正式N256成绩。

原始根目录`/workspace/benchmark-results/20261020_memory_combo_service4k_tianshu`。关键独立复核见DIAGNOSTIC_4K_GATE_32g.json、DIAGNOSTIC_4K_ROUNDS_32g.csv。旧verify_service.py和失败pair入口保留，新的verify_screen_diagnostics.py明确不同的诊断口径。
