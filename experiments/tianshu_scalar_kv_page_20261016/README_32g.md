# 天数 BI-V150：Decode 页表一次读取

当前原始命令阶段计时中前向占88.48%，已有当前.cv attention微测约.846ms/层，而旧四类GEMM合计约.196ms/层；这些来自不同测量，不能直接作为完整Graph算子百分比。

本实验生成独立.cv副本，加入SCALAR_KV_PAGE constexpr。在3D且TILE_SIZE==BLOCK_SIZE时，seq_offset=j*page_size+[0,page_size)的页号恒为j，故只读一次block_table页号再广播；其它情况回到原始读取。不改int64整数/64位指针、浮点算术、reducer、tile/warp/stage、grid、.cv及S8。False模式为同生成源码的恒等对照，True模式才启用标量读取。源码两处替换可逐字逆还原8deb，CPU66810个映射检查通过。

GPU实验尚未运行。计划17场景/34候选：当前各batch的4K，B64的5K非连续和65长度padding，以及B8的10个页边界；同一输入、原输入与变Q/K/V Graph、GPU FP32/独立CPU抽点、尾部、实际metadata和9轮交替event。B24仍KV32/w2所以标量守卫不启用，作为回退检查。不因局部测试通过修改最佳启动器；先有明确C64收益再考虑服务对照。

运行：`python3 experiments/tianshu_scalar_kv_page_20261016/run_probe.py --output NEW_ROOT --stop-at-utc ISO`。父进程冻结10份源码，GPU子进程串行、限时，结束检查GPU空闲并自动备份。当前正式4K3992.97、Math103/105不变。
