# 原生weight-only真实Down/M64：数值通过，性能严重负向

Torch原生aten::_weight_int8pack_mm在小尺寸BF16契约12组通过后，进一步测真实checkpoint第0层Down2048x6144，M64，NT/NN布局。用已合格的真实fused GateUp64产生BF16隐藏态H，含整个MLP或仅Down；返回分配直接捕获，控制和候选均无额外输出复制，包含原生API内部转换/分配。七轮轮换顺序、每组20replay，Graph/eager均测。真实服务BF16是vendor4c2e23，M2048特例hook对M64明确委托，torch.mm仅作次对照。

|Graph范围|服务BF16 ms|原生NT ms|原生NN ms|
|---|---:|---:|---:|
|仅Down|0.05939558|9.92799530|10.99820633|
|整个MLP|0.11826116|9.86061783|11.06812668|

单Down原生NT耗时约167倍；不同scope的微小计时次序差异不影响巨大负结论。不能从API支持推断低比特计算已加速，也不凭此实验断言底层指令或唯一瓶颈。单层cache screen已经负向，不做全42/Math或formal，不采用。

两组全新输入seed202610239200/201，完整GPU eager以及wholeMLP/Down两个captured-returned-outputs poison重放均通过，A/H/工作缓冲守卫及RNG、GU精确binary、所有权重pack/scale/checkpoint/source/provider哈希不变。独立CPU原始Down重pack和全部524288候选输出FP64数学界通过，NT/NN逐位相同，BF16控制完整数学界通过；声明gamma(2K+4)+FP64误差+半BF16ULP，不冒称FP64逐位精确。

原生大库只SHA、未复制上传；914源+CPUgate首GPU前冻结，GPU后零编辑。最终硬件GPU空闲，完整raw数组/日志在容器，GitHub仅源码/报告/Gate/关键CSV。发布r3仍5950.75tok/s。该结论只涵盖此真实M64 DownAPI实现，不证明所有W8A16或其他M/卡都慢。
