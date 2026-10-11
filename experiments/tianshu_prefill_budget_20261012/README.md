# 天数Prefill token budget短扫（尚未启动）

前提是已验证的Native Decode + FA2 Prefill配置。当前服务max_num_batched_tokens=2048，4096输入需要至少两个chunk。假设把budget扩到4096或8192能减少重复执行和metadata开销；代价可能是单步阻塞更长、Decode TPOT及KV容量变化。

计划只变更一个CLI参数，保持模型、采样、Attention、dtype和原始benchmark脚本。先用4096输入/128输出、16并发/16请求、五轮跳首轮筛选，记录TTFT、TPOT、实际KV容量、greedy输出及失败；有收益后再测4096/1024较大批次。不把自定义短实验当作官方成绩。新计算批次范围须保存实际导入路径与独立质量验证。

本文件只是待验证假设，没有声称已有收益，当前GPU仍在Decode自适应策略A/B。
