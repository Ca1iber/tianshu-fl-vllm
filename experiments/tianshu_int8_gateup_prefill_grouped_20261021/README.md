# 大 M INT8：program 顺序的 L2 复用候选（仅准备）

当前 INT8 GEMM 的 program 按 M 外层 / N 内层逐行块扫描。大 prefill M 的每个行块会重新扫整份 W，故增加标准 grouped-M（8 个行块）program 顺序，与 group1 逐项比较，目标是增加权重 L2 复用。这是动机，没有硬件计数器证明，不预设收益。载入布局仍是 NN，量化、整数点积、scaling、mask 不变。

六个原分块各 group1/group8，共12配置；M128/129/256/512/1024/1152/2048/4096，含部分行 tile 和最后不足8个 M tile 的 group。CPU仅证明三处可逆源码编辑、量化 AST 不变、480组几何一一覆盖且边界不越界；GPU每个配置仍需独立整数/数值/RNG/guard检查。基线保持当前 BF16 linear_contiguous；group1与group8同组仅执行顺序变化，静态 W 打包除外、A量化计入。

等待当前 row-major prefill 组件完成后再决策启动，未排队、未启动GPU、未模型、未新Math/正式测试。完整原件只容器，关键结果推GitHub；23:58北京时间硬停止。
