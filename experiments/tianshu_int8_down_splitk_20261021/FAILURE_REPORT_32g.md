# 第一版SplitK在首次tail编译失败

GPU结果phase=failed、complete=false，尚无任何tail/真实Down数值或性能结果。异常位于producer的range(chunk//BK)：在内核内通过tl.cdiv创建的constexpr长度被后端当作需要_semantic的运算处理。所有原始源/CPU gate/失败日志保留，不修改本目录已冻结的实现。

新v2目录将CHUNK改为实际CPU helper计算并作为constexpr参数传入，CPU覆盖验证也直接调用同一helper；不改变矩阵分段、量化、数值门或计时范围。旧版CPU检查通过不代表GPU编译/数值通过，不能据此声称性能收益。
