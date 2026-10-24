# 每行倒数替代P逐元素除法：全部数值通过，完整组件仍无收益

2026-10-24，BI-V150，GPU PID28615已complete并释放。保留scale=div_rn(peak,127)，只将P量化由nearbyint(div_rn(weighted,scale))换成r=div_rn(1,scale)后nearbyint(weighted*r)。同页表、K8、动态V8 writer/scale、8448B真实分页、QK、online softmax、reducer；两整数组均S4/T32/W1/stage1。r5与checkpoint未改。

|B64热Graph|当前K8/V BF16 ms|原整数PV ms|每行倒数PV ms|
|---|---:|---:|---:|
|4K上限，实际4089|0.585798264|1.217440605|1.218210983|
|5K上限，实际5113|0.724416351|1.514874363|1.515896893|
|ragged/strided|0.576767111|1.091665745|1.093307304|

9轮正反交替×20重放，包含真实量化writer、K/V scale写入及完整reader/reducer，无本代理后台CPU准备。候选相对原整数版+0.0633%、+0.0675%、+0.1504%，视为无收益；相对当前K8/V16增加107.96%、109.26%、89.56%。没有≥5%合格固定配置，不接engine，不跑Math或正式benchmark。整数PV这版实现停止推进，不能泛化为所有INT8 attention都不可行。

源码逆变换证明只改变P的除法求值顺序。CPU2681个不同数量级的舍入边界/相邻码检查中35个量化码与严格版不同，明确为新的FP32双舍入合同，不宣称逐码等价。两公式均满足预定实数误差≤.5001scale；zero/tiny保持有限。实际GPU再对严格/倒数各随机、变数据、ties邻居、极小值、全零五相位验证，共5120个Pq、20480个INT32和20480个FP32输出字；保存的原始P/V/scale由独立标量FP32和Python整数点积重算，逐字通过。

三完整场景、同Graph两阶段改变Q/K/V/scales/table/lengths/slots，2359296个GPU FP32参考输出及27648个所选CPU FP64参考标量均符合原.002+.01abs门槛。完整raw缓存、负slot、所有输入、输出/实际分段scratch guard和CPU/GPU RNG保持；独立复核196608个实际writer V整数、1536个Vscale和1536个Kscale字完全一致，所有源/快照/真实编译文件/原始9次event中位数均通过。失败前保存模块已加入，但本轮没有数值失败；它不补写上一轮缺失的失败输出。

实际资源两整数版均254regs/0spills/shared8704；control142/0/8192。选定同一连续case编译文件全文件静态计数：LLVM fdiv12→8、ASM rcp4→4、FMA68→60、INT8矩阵指令8→8，见STATIC_CODEGEN。代码确实不同且运算减少，性能没有兑现；没有动态计数器，不将其解释成已经定位其它瓶颈。

原始完整数组/源码副本/IR/ASM/event/日志留容器，持久GPU及CPU压缩档均逐文件tar compare通过，路径/大小/SHA见RAW_ARCHIVE。代码/计划/关键CSV和GATE/报告推天数GitHub。当前正式4K最好仍r5 6123.31total/1224.66output tok/s、TTFT3202.70ms，本轮没有新服务成绩。
