# Down INT8 split-K：准备中，尚未执行GPU

现有Down组件在M4096的完整A8量化+INT8 GEMM只有1.102635→1.038863ms，收益不足，decode更慢。新假设仅在同样的整数运算中拆分K6144，尝试提高并行度。producer写独立INT32 partial，reducer用INT32确定性求和后一次乘原scale；不用浮点atomic，不改舍入或量化规则。所有中间写入/读取、A动态量化和归并都包含在热图计时；静态W打包/额外workspace单独记录，不能由组件结果声称模型收益。

预先固定9组配置，真实Down M129/1024/2048/4096；保留原BF16 mm（含现有M2048特殊配置）、辅助BF16 linear、原始最佳M4096 unsplit INT8作为同次对照。每个组合三阶段改变输入/权重符号/scale，完整INT32 GPU对照在实际绝对乘积和<2^24时与FP32整数参考逐位相等；CPU独立重算所有A量化、checkpoint选中原行、选中输出INT64以及每段partial。另用M3/17,N130,K259、strided输入和零/RNE固定值，54份完整小图输出/partial在CPU重算。

7轮四路旋转次序，每轮20次图重放；保留全部编译失败。最大workspace split6×4096×2048×4=201326592字节；这是单算子workspace，不推算模型显存安全。未接入模型；Down INT8精度也尚未通过新Math。若组件收益足够，模型接入仍需真实allocator/native显存计费、独立105题Math>95%、4K服务对照。

当前TopP候选Math/正式4K优先，全部退出并确认GPU空闲后才可启动本实验。只用天数容器，23:58北京硬停，不修改任何已经GPU运行的旧实验、推荐版、安装源或沐曦仓库。
