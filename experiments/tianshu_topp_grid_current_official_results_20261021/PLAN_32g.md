# 当前TopP16K/grid2正式4K归档：预先固定比较标准

尚无本目录新正式成绩。Math与官方4K均须完整结束，原版13项serve、438运行源、62环境键和13项控制器源完全绑定；重新核对原始CSV/stdout、105题独立新Math>95%、真实C8/C64新采样与原有INT8/KV/图/抢占审计。不复用407源旧正确率，不测试完整16K。

正式4K使用原官方脚本C64/N256/4K/1K四轮，剔除首轮；历史参考为已通过的407源5586.28 tok/s、TTFT3780.63ms。历史参考不是同次AB，对照小测已单独完成+0.83956%。在取得新正式结果前固定推荐条件：相对当前正式吞吐提升>0.5%，且大于双方各自去首轮波动；新波动<0.5%，TTFT不超过此前正式值1%，无API就绪后现场调优、无抢占/虚拟KV泄漏，满足官方公开4K门槛。没有达到推荐条件也保留完整新结果并明确原因，不改动原始比较或重跑刷数值。

CPU归档命令：`env -u PYTHONPATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_topp_grid_current_official_results_20261021/verify_official.py /workspace/benchmark-results/20261021_topp_grid_current_official4k_tianshu /workspace/benchmark-results/20261021_topp_grid_current_math_tianshu`。
