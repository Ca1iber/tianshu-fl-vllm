# 整数P×V完整attention：两场景性能负，第三场景误差门禁失败

2026-10-24，BI-V150，PID27458已结束、GPU空闲。本轮补全21日原语之后缺失的writer+分页attention实验，以当前qualified S16标量K读取SHA7b94...为控制，未使用旧paired-K基线。原r5模型/checkpoint和6123.31正式4K不变，没有engine/Math/服务或新正式benchmark。

|B64实际热Graph|4K上限，实际4089 ms|5K上限，实际5113 ms|
|---|---:|---:|
|当前K8/V BF16，S4/T16/W1|0.585605192|0.724803257|
|动态V8，浮点PV，S4/T16/W1|0.850531197|1.055426788|
|动态V8，浮点PV，S4/T32/W1|1.545095158|1.923948097|
|动态V8，整数PV，S4/T32/W1|1.214989185|1.510983276|
|动态V8，整数PV，S4/T32/W2|1.472139645|1.821466255|
|动态V8，整数PV，S4/T64/W2|2.637316132|3.275839233|

9轮正反交替×20重放，包含真实V动态量化及K/V scale写入。最快整数配置T32/W1比当前K8/V16延迟增加107.48%/108.47%，即约2.075/2.085倍耗时。相同T32/W1下整数版比有spill的浮点V8快，并不能作为相对现用路径的收益。计时中无本代理后台CPU准备/归档/推送。

第三个ragged/strided场景在T64/W2出现运行期数值失败：3个元素超过预定 .002+.01*abs(reference)，最大绝对差0.0024836454540491104。按原门禁退出，未测第三场景耗时，未改阈值、未删失败配置、未重跑。原result保持complete=false/phase=failed。失败输出数组未在assert之前保存，因此3个差异数量来自原运行期断言/日志，不能声称对该失败输出作了独立CPU重算；完整审计不合格。后续新探针应在失败断言前保存数值输入/输出，这不追补本轮原始证据。

独立CPU复核确认：两完整场景共3145728个GPU FP32参考输出、36864个所选CPU FP64参考标量的原运行记录、所有输入/raw缓存/输出/实际分段scratch/RNG保护，以及33份源/快照、编译IR/ASM和原始9组event中位数均符合冻结协议。原整数原语重新验证T32/W1、T32/W2、T64/W2各两相位；保存的4096个Pq、12288个INT32和12288个FP32输出字由独立标量FP32与Python整数点积重新计算，逐位通过。24份实际writer样本另重算393216个V整数值、3072个Vscale和3072个Kscale字，全部通过。上述局部通过不改变完整运行失败和不采用结论；PARTIAL_CPU_GATE明确所有范围。

原生页8448B=K4096+V4096+Kscale128+Vscale128，全部scale计入FullAttentionSpec.page_size_padded并别名同份raw；相对12288B理论页字节减少31.25%，但本轮未接全engine allocator，因此没有实测全模型KV容量/显存结论。原K软件8/QK/online softmax分母/reducer不改，整数PV把Vscale并入P再逐query/tile量化，构成显式精度变化，需要独立模型质量，未借用旧Math。

编译资源：control142regs/0spills/shared8192；V8浮点T16为177/0/8192，浮点T32为256/18/12288；整数T32/W1为254/0/8704，T32/W2为212/0/8704，T64/W2为256/99/13312。n_spills按运行时原单位记录。真实ASM包含ml_matrix_mad_i32_i8_m16n16k32；LLVM有P量化fdiv，ASM有rcp及FMA精度修正。静态证据见STATIC_CODEGEN，不能替代动态硬件counter，也不把全部负收益归因spill或除法。可另立候选隔离P量化的逐元素精确除法成本，改用每行倒数后乘法时须声明新舍入合同并重新验证。

代码、计划、独立partial核验器、关键CSV/JSON及本报告进入天数GitHub；完整原语/实际writer数组、源码副本、IR/ASM、失败日志及原event保留容器，持久压缩档已逐文件tar compare，路径/大小/SHA见RAW_ARCHIVE_32g.txt。原完整通过审核器保留，不修改它来让本次失败冒充成功。

对照范围补充：本实验使用7b94历史合格标量K读取器作组件控制；正式r5使用f1df成对K读取，不能将本组件ms等同r5实际attention成本。详见[当前KV基线说明](../tianshu_framework_status_20261024/CURRENT_KV_BASELINE_NOTE_32g.md)。原始数据、门槛及未采用状态不改。
