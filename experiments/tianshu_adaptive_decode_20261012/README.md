# 天数自适应原生Decode实验

基线为最佳tag tianshu-best-hybrid-20261012：正式4K2286.09 total tok/s、TTFT7569.57ms、MATH100/105。旧Hybrid helper SHA保持不变，新增独立开关VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS，默认0。

独立80配置均通过固定FP32、CPU参考及共享S64缓冲区未用tail检查。策略为批次1→S64、2–15→S16、16–64→S4；按GPU cu_seqlens的静态形状选择，适配Graph padded batch，不根据运行时GPU值同步或临时调优。Builder scratch一次按S64分配，两个原生kernel用选定分段的packed前缀。额外约24.4MiB内存需在服务实测KV容量。

真实metadata/Hybrid/Graph联合13场景通过：Prefill与回退不变，B4/B8保持原生分段，B64约快7.9%，B1/32768历史约2.88x。没有沿用旧MATH作为新策略准确率。

run_service_series.py 串行比较两组：Hybrid+S16与Hybrid+adaptive，4096输入/1024输出、64请求/64并发、三轮跳首轮；另每组三次4096输入/4096输出、单路生成诊断。全部属于缩小/自定义实验，不能冒充官方256请求均值。run_math_eval.py / audit_math_eval.py 对新路径独立评测105项，并记录实际adaptive源SHA。

每个实验记录源码、环境、数值、服务/数学结果、失败样本与采用结论。短服务A/B已完成：64路4K/1K、64请求三轮跳首轮，2322.92→2398.00 tok/s（+3.2322%），TTFT23186.42→23326.66ms（约+0.6%）；单路4K输入/4K输出67.8360→56.1689s（−17.1990%）。404请求全部成功、KV容量均521936。不冒充官方成绩。新数学正在独立评测，未采用候选。GPU串行，只清理自有进程，截止2026-10-12 11:58。
