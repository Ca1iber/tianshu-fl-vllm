# 天数 BI-V150：Decode 页表一次读取

当前原始命令阶段计时中前向占88.48%，已有当前.cv attention微测约.846ms/层，而旧四类GEMM合计约.196ms/层；这些来自不同测量，不能直接作为完整Graph算子百分比。

本实验生成独立.cv副本，加入SCALAR_KV_PAGE constexpr。在3D且TILE_SIZE==BLOCK_SIZE时，seq_offset=j*page_size+[0,page_size)的页号恒为j，故只读一次block_table页号再广播；其它情况回到原始读取。不改int64整数/64位指针、浮点算术、reducer、tile/warp/stage、grid、.cv及S8。False模式为同生成源码的恒等对照，True模式才启用标量读取。源码两处替换可逐字逆还原8deb，CPU66810个映射检查通过。

GPU实验已完成并本地离线核验：17场景/34候选全部通过数值、原输入及变Q/K/V Graph逐位对比、CPU抽点与尾部保护。459个event及10份源快照已核验。B64/4K耗时+0.02355%，5K+0.00138%，没有收益。B64实际ELF的.text在原版、恒等控制和启用标量读取间逐字节相同：11272bytes，SHA03685e85…。LLVM重复i32读取静态调用从36变5，但后端最终机器码相同；不能把LLVM变化当作执行变化。

不接入服务、不新跑Math或正式benchmark。完整结论、数据与复核命令见[REPORT_32g.md](results/20261016/micro_32g/REPORT_32g.md)。

运行：`python3 experiments/tianshu_scalar_kv_page_20261016/run_probe.py --output NEW_ROOT --stop-at-utc ISO`。父进程冻结10份源码，GPU子进程串行、限时，结束检查GPU空闲并自动备份。当前正式4K3992.97、Math103/105不变。
