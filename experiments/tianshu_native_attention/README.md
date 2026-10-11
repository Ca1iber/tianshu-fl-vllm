# 天数 BI-V150：Native Decode + FA2 Prefill

当前已验证优化在单卡BF16 MiniCPM5-2B保留原生3D Decode，将Prefill和mixed batch拆分并使用FlagGems FA2。通用源码保留opt-in；天数推荐启动脚本显式开启两个已验证开关，保持官方serve参数。

正式4K（4096输入/1024输出、64并发、256请求、四轮跳首轮）1024/1024成功。均值2286.09 total tok/s、457.22 output tok/s、TTFT7569.57ms；本机原版1876.67/12315.86，吞吐+21.8163%、TTFT降低38.5380%。公开基线2028.01/11573.53，吞吐+12.7258%。三轮波动跨度5.6616%，完整原始CSV保留，不额外剔除第二轮。

独立MATH-500 Level3完成105项、无请求/评分错误，100/105=95.2381%通过用户给出的官方>95%门槛。原版101/105仅作对照，不作为否决条件。原始错误、两题max_tokens及数值失败配置都保留。

运行：bash /workspace/vllm-plugin-FL/experiments/tianshu_native_attention/serve_tianshu_minicpm.sh。脚本设置HYBRID_ATTENTION=1、COMPACT_VARLEN=1，旧全FA后端、split、75点padding与logits gather为0。本轮helper SHA256为5c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4。没有修改vLLM核心安装源码或重排权重。

本目录包含真实3D Decode分段/tile短扫、服务profile、独立FP32+metadata+CUDA Graph验证、缩小负载A/B、独立数学审计和原样官方4K进程管理。results/20261011保存正式汇总、原始CSV与精简来源证明；完整材料与报告位于工作区service-results/20261011_native_decode_fa2_prefill_report.md及对应实验子目录。

每项优化须独立写报告，记录假设、代码版本、实际导入路径、质量与性能范围、失败记录、原始数据和采用结论。GPU实验串行，仅清理自有进程。用户已预约至2026-10-12中午12:00，停止截止11:58；当前最佳版本保存后继续用短实验探索，未授权完整16K。
