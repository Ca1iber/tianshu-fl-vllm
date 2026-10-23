# RMS与LM组合正式4K独立CPU核验

重新检查完整原4轮CSV每轮256请求、请求/生成token量、排首轮统计、全量模型/控制器快照和原checkpoint SHA，逐字段比较所有原始comparison。重新执行全部继承与实际RMS/LM运行审计，检查42层norm/GU capture及二进制、LM新计数/实际M64/CUBIN/原pack/Profiler峰值，旧独立A8/旧LM64不声称。与该组合首次新鲜C8/C64各105题的独立门槛绑定，同期短4K必须先通过。

只有完整原始正式成绩>r3的5950.75×1.005、热轮spread<1%、TTFT<=3174.58×1.01、全部请求成功、零preemption/ready后autotune，才输出promotion_passed=true；本核验不自动发布。没有运行GPU前结果或采用承诺。原始文件不改，CPU过程不导入Torch/Triton/vLLM/FlagGems。独立比较完整JSON仅压缩摘要存GitHub，数组/响应/日志只容器。
