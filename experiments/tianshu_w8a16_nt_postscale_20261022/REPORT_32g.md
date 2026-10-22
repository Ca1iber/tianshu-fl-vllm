# W8A16 NT后缩放：更接近但仍不采用

全部15配置/45相位及独立CPU审核通过，原checkpoint与源不变。整数NT权重布局、activation BF16、dot后每通道缩放，两个2warp小tile和一个4warp tile作为整体候选；CPU独立审核9个完整decode、29259个FP64 selected输出、85950027个全输出cast元素。

|角色/M|tile|BF16 ms|W8A8 ms|候选 ms|候选/BF16|
|---|---:|---:|---:|---:|---:|
|down/8|2|0.092910|0.135936|0.104304|1.12x|
|down/64|2|0.059332|0.069455|0.226201|3.81x|
|lm_head/8|0|0.820179|0.873838|0.982689|1.20x|
|lm_head/64|2|1.031424|0.613233|2.779977|2.70x|

7轮轮换顺序，每轮20次热Graph，中位数取预定tile最小值。所有实际形状仍慢于BF16，M64 LM还慢于现用W8A8。NT布局和tile同时改变，不能把综合变化归因于布局单项。raw `/workspace/benchmark-results/20261022_w8a16_nt_postscale_GPU_tianshu`。没有新Math或正式benchmark，不采用。现行正式发布保持5695.95 total tok/s。

已准备在NT整个队列通过审核后用IXKN读取实际kernel指令/访存/占用率；此时没有实际计数器结果，不能从寄存器数或时间单独判断根因。
