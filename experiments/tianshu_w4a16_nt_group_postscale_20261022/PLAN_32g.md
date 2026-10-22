# W4A16 NT布局与分组后缩放（CPU准备）

独立候选把 packed 布局由 [K,ceil(N/2)] 改为 [N,ceil(K/2)]，相邻 K 两个 nibble；同一 K128 分组内先点积，再乘 FP32 scale。pack_nibbles_nn(q.T) 得到此 NT 布局，完整 decode 要重新验证；原 checkpoint、整数码、分组量化数学定义不改。

真实 Down / LM head 的 M8/M64，tail M17/N131/K259。三个预定 tile 分别 BM16/BN64/BK128/warps2、BM32/BN128/BK128/warps2、BM64(若M64否则16)/BN128/BK128/warps4。布局与 tile 是整体候选，不把综合差异归因于单项改动。三输入相位、完整 GPU 参考、独立 CPU FP64/format decode/cast/INT8对照、7×20 Graph 计时与前版一致。

本目录此时仅 CPU 准备，不在现有队列中。GPU 仅在之前作业结束且独立审核完成、卡实际空闲时启动。新的数值顺序不能借用现行 Math；阳性组件才考虑引擎/内存/新Math。每任务最多30分钟，每日北京时间06:00–次日00:00，本日23:58硬停止。原件不修改、raw仅容器，源码报告关键数据放天数 GitHub。
