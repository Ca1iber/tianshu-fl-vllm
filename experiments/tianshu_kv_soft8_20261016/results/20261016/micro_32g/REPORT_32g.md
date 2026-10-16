# 软件8位KV：最快仅省约1%，资源代价明显

手工页交错uint8有限E4M3/scale1over16，整数位操作还原BF16；非原生FP8硬件或vLLM KV dtype。当前producer/reducer AST不改，只新增_uint8、KVMode0的cast helper分支，BF16完整对照保持相同。

|配置|4K ms|变化|5K ms|变化|regs|spill|
|---|---:|---:|---:|---:|---:|---:|
|BF16|0.845438|+0.000%|1.052645|+0.000%|85|0|
|Soft8_t16_w1_s8|0.837867|-0.896%|1.038083|-1.383%|244|0|
|Soft8_t16_w2_s8|1.465380|+73.328%|1.820336|+72.930%|221|0|
|Soft8_t32_w2_s8|2.895335|+242.466%|3.598338|+241.838%|256|125|

最快tile16/w1在4K-0.896%、5K-1.383%耗时，244 regs对85，无spill；tile32/w2达256 regs/125 spill且耗时约为BF16的3.42倍。没有计入量化写入、完整allocator或校准，不因半存储量声称服务提速，不做模型接入/新Math/正式，未采用。下一项只测试相同格式下uint16→uint32解码中间量，不能先断言高寄存器就是瓶颈。

自定义格式指数0的编码全部flush signed zero；其他所有指数有限、最大幅度30，BF16输入RNE丢4尾数位并饱和。固定scale在随机fixture未clip，不能推出真实42层无需校准。近似误差对原BF16 FP32参考单列，自身正确性对实际反量化BF16的FP32参考，不复用当前Math为该编码质量结论。

本地独立核验65源快照与所属源、生成源重建/producer与reducer AST同源、全部65280 finite BF16 CPU编码与独立frexp/round参考、全部256个实际GPU解码值含signed zero与CPU逐项相同；4阶段改变Q/K/V清输出重放、16项自身FP32和96项CPU head检查、尾部、56个交替计时组/中位数及差额通过。未序列化完整GPU Tensor，只重验记录和出处。

归档4,351,761字节、17分片，SHA256 18307ec62b1924da588a4d12c11abf34ed9111bc18b977e5d16048d0b330a8ec。原件service-results/20261016_tianshu/kv_soft8_micro/20261016_kv_soft8_micro_tianshu。当前正式4093.81/Math102不变。
