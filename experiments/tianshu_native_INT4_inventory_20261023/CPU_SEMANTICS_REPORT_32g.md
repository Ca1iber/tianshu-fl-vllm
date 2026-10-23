# 独立CPU实现确认通用INT4量化语义

未改首次GPU前冻结的4例输入/代码/预期。独立使用原生CPU的_convert_weight_to_int4pack_for_cpu与_weight_int4pack_mm_for_cpu，q=7/8/9/10、scale1/8、zero0、group128，输出分别-4/0/4/8，65536格全部精确一致，确认signed center8语义。CPU pack布局与CUDA Meta不同，未据此推断CUDA打包或硬件已可用。

GPU微型测试仍排队，性能和真实权重正确率均未测；原EXECUTION所有SHA保持。reference_cpu.py及CPU_SEMANTICS记录独立CPU证据，不替代GPU结果或改变GPU预期。
