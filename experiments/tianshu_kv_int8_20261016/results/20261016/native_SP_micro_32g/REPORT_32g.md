# 天数逐token/head INT8 KV：attention首轮更慢

原生支持int8_per_token_head。此处仅手工量化fixture后测attention+reducer，刻意不含真实KV writer或量化成本，也没有整模型/Math。保持.cv、page16、S8、原Q与BF16控制：比较对齐packed128+独立FP32 scale、原生inline132 scale和32tile/warp2；没有将dtype开关丢失现有路由后的结果作为比较。

3场景、6阶段均通过各自dequantized FP32及CPU抽点参考。原地址改变Q/K/V及scale，先清空output再重放Graph，输出/segment尾部通过；另外记录对原BF16输入FP32 attention的近似误差，不假称INT8与BF16精确等价。Tensor全量未序列化。

|场景|BF16 ms|INT8 packed128/w1 ms|变化%|INT8 inline132/w1 ms|变化%|
|---|---:|---:|---:|---:|---:|
|B8/33|0.015913846|0.016986538|+6.740626|0.017287500|+8.631821|
|B64/4096|0.849829197|0.909797287|+7.056487|1.021284580|+20.175276|
|B64/5120|1.057626629|1.131307507|+6.966625|1.268218136|+19.911706|

B64/4K packed128也慢7.0565%，inline132慢20.1753%；packed128/tile32/w2慢11.9549%。均未计写入量化成本，当前无进服务的收益依据，不采用。B64控制n_regs85，INT8三配置164，均n_spills0；这只是静态资源记录，未采硬件counter，不据此断言唯一瓶颈。

62份冻结源及自有文件SHA、84个原始event、3案例/6阶段记录、实际quant_mode/stride/grid/tile/warp均本机复核。原包4199705bytes，SHAfc06cf07e2b1ff73f0749ba00797e551227a3cee254aa9634cca53822c7887af，原件/workspace/benchmark-results/20261016_kv_int8_attention_tianshu。`python3 experiments/tianshu_kv_int8_20261016/verify_downloaded.py --root <下载目录>`。

后续独立变体测试将KV在load时反量化为BF16，再走普通QK/PV而不在S/P上乘scale；这会增加BF16舍入，仍需独立FP32/CPU检查和新Math，当前尚无该变体收益。正式最佳4093.81/Math102保持，无新16K。
