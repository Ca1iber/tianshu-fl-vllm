# LM64原生INT8：无组件收益，维持现有实现

当前LM64是130560x2048的W8A8，Triton INT32 GEMM内完成缩放与BF16写出。新候选使用同一NT bank的转置view、torch._int_mm和单独反量化，没有新布局或权重副本。

8组BLOCK/warps、7组交错计时全部完成。eager基线0.621214ms，最快B512/W4为0.623298ms，组件加速-0.3343%；Graph基线0.616979ms，同配置0.621215ms，-0.6820%。其余更慢，不接入服务，不跑完整benchmark。正式推荐版5777.22 tok/s保持不变。

拆分计时：原生int_mm 0.521036ms，比原有GEMM含反量化0.614415ms快；但独立反量化最快约0.095494ms，加上A8量化约0.005653ms，整体没有收益。分项相加仅作解释，以完整路径交错计时为准。

三阶段改变A/权重符号和scale，所有8候选完整INT32与BF16 logits均与当前实现逐位一致；图回放前污染输出/INT32，结果与eager一致。独立CPU重算完整W8、A8、scale，1,824个实际INT64/投影输出一致；原checkpoint前后SHA256相同，899份源/控制输入冻结不变。M8保持BF16未测试，未声称新的Math质量或整机吞吐。

源码、计划、CSV、分项及独立审核gate存天数GitHub；raw仅容器 /workspace/benchmark-results/20261022_native_int8_lm64_GPU_tianshu。
