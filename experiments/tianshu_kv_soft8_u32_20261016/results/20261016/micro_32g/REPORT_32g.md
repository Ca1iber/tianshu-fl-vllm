# 软件8位KV整数中间量：u32更快，但仍仅组件

同一轮BF16与相同有限E4M3/scale1over16编码对照，仅把解码中间量uint16改uint32，最终uint16→BF16 bitcast保留。producer/reducer AST与原同源，旧codec SHA固定；不是原生FP8、没有writer/完整allocator/校准/模型/Math/服务成绩。

|配置|4K ms|相对BF16|5K ms|相对BF16|regs|spill|
|---|---:|---:|---:|---:|---:|---:|
|BF16|0.846088|+0.000%|1.052185|+0.000%|85|0|
|Soft8_u16_t16_w1_s8|0.838008|-0.955%|1.038148|-1.334%|244|0|
|Soft8_u32_t16_w1_s8|0.820979|-2.968%|1.016194|-3.421%|246|0|
|Soft8_u32_t16_w2_s8|1.416916|+67.467%|1.759467|+67.220%|222|0|
|Soft8_u32_t32_w2_s8|2.766683|+226.997%|3.431139|+226.097%|256|124|

最快u32-t16/w1比BF16省2.968%/3.421% attention组件时间，比同次u16约再省2%；regs244→246而非减少，未支持“降寄存器就是增益原因”的解释。两版同tile/w1四阶段输出逐位相同，实际cubin ELF可执行.text段在两场景均不同，不是单纯源码差异被编译器完全消除。尚未取硬件计数，不断言具体pipe/occupancy瓶颈。

每配置首次warm launch保存source/TTIR/TTGIR/LLVM/设备汇编/cubin，发生在事件计时前，Graph replay中没有文件I/O。十组metadata共60个编译文件逐字节SHA/长度核验，解析ELF32/64可执行.text与各自二进制对应。全部68源/生成源、256个实际GPU码含signed zero与独立CPU参考、65280 finite BF16编码、4个改变QKV阶段/20自身FP32/96 CPU head/尾部与70个事件组独立核验；完整GPU Tensor未序列化，不声称本地重算整个attention。

目前仅3%左右组件收益，写入/校准/真实cache生命周期未计，尚不进入正式最佳；真实端到端是否有收益仍需框架接入和新105题严格>95质量，不能借用现有102/105。当前最佳仍4093.81。

归档5,273,983字节、21分片，SHA256 c2046b1fc1dae03976e071935416baa32aa3dd48021bbb9c588cb3bba180c731。原件service-results/20261016_tianshu/kv_soft8_u32_micro/20261016_kv_soft8_u32_micro_tianshu。
