# 天数Prefill预算实验

短扫完成：4K输入/128输出、16并发/16请求、五轮跳首轮，2048/4096/8192分别4585.15/4895.48/5004.03 total tok/s；4096+6.768%，8192+9.135%。261请求全部成功，7个固定greedy探针一致。8192客户端TPOT−12.104%，TTFT+0.186%，实测KV容量−0.29%。这些是缩小实验数据，不是官方成绩。

实际Worker/Hybrid/adaptive源码三组完全一致，保持冻结。8192独立数学已完成102/105（97.14%），105请求/评分齐全、无错误，全部正常stop；结果没有复用2048分数。正式4K正在独立目录20261012_budget8192_official4k_tianshu运行，尚无最终成绩。run_budget_pair.py已开始2048/8192的4K/1K、64并发/64请求、三轮跳首轮对照；有收益再用run_math_eval.py独立评测新预算，audit_math_eval.py核验严格>95%及实际预算/源码。run_official_4k.py要求有清晰短测收益、对应源与预算的质量通过、未修改的官方脚本SHA，才运行256请求四轮跳首轮。

完整报告和摘要见results/20261012，原始数据位于/workspace/benchmark-results/20261012_prefill_budget_tianshu及本地service-results/20261012_prefill_budget_tianshu。GPU性能任务串行，只清理自有服务，11:58停止GPU工作。
