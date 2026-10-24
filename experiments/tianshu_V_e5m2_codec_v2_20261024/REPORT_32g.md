# V改为E5M2后decode组件耗时增加，不接入服务

BI-V15032G，保持当前K软件E4M3、Q/K BF16、softmax、S16 reader和现有launch/reducer；V改为E5M2 byte，实际组件page12288→8192字节。BF16_PV解码回BF16，FP16_PV直接用FP16 V与P×V；后者P精度变化单列。两者都不修改checkpoint，不依赖原生FP8计算。

v1因FP16 helper回退return类型合并失败，原错误与源码另存且未产生计时；v2将回退写成显式else，重跑CPU65536位模式和完整GPU证明，保留v1原始失败。v2所有65536种BF16 V编码与独立CPU参考一致，K编码全部不变；两种reader的248有限码值/符号、8非有限分类通过，writer不产生保留码。

10个batch/长度/strided场景×3路径×2个同Graph变数据阶段，3465216个完整GPU FP32参考输出元素、79872个所选CPU FP64标量检查通过。实际writer、分页/负slot、全部raw storage、输出/scratch、全输入字节与RNG检查通过。Q/K/V、页表、lengths、slots在第二阶段原地变化并重放同Graph。每组9次交替顺序×20 replay；计时包含一次实际KV更新和单层decode，参考计算在计时外。

|B64组件热Graph|当前K8/V16 ms|V E5M2→BF16 ms|耗时变化|V E5M2直接FP16 ms|耗时变化|
|---|---:|---:|---:|---:|---:|
|4K上限|0.589286|0.836469|+41.95%|0.777360|+31.92%|
|5K上限|0.728894|1.041681|+42.91%|0.960842|+31.82%|

第二变数据阶段长度减7，实际计时长度4089/5113；两组输入对应相同。其余8场景两种候选也全部更慢，全部记录见COMPONENT_KEY_TIMES_32g.csv。负的time_reduction不是同幅度的吞吐下降，不能把组件百分比换算为服务吞吐。

B64实际配置同为tile16/warp1/stage1/segments4，寄存器142→156/154，spills全部0，共享内存8192→6144B。未测achieved occupancy或HBM分区计数，不据寄存器单一指标断言慢因。内存占用减少并没有带来这组实际decode提速。

独立verify_gpu_cpu.py核对全部冻结来源/快照、原始event中位数/变化率、数值与保护记录；没有序列化全部Tensor，不声称离线重新计算完整attention。没有候选满足预声明B64/4K与5K都减时≥5%的条件，所以不接引擎allocator/prefill、不跑新Math或完整服务benchmark。正式r5仍为6123.31 total tok/s，本组件不改r5。

下一项只排查物理page步长：对同一FP16_PV码流/reader/launch分别用8KiB与12KiB页面，与当前K8/V16比较，先隔离布局因素，不预设能改善性能。尚无该排查结果。当前全部CPU/GPU原始生成源、日志与结果保留容器并归档比较，路径与SHA见RAW_ARCHIVE_32g.txt；GitHub仅保存代码、报告和关键结果。
