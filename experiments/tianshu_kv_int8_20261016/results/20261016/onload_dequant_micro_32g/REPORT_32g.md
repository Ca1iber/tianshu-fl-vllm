# 天数INT8 load时反量化：更慢，不采用

独立变体保留原.cv模板和INT8 S/P scale路径，4处可逆生成；DEQUANT_KV_ON_LOAD只对mode2生效。先将int8 K/V乘各token/head scale并舍入BF16，再走普通QK/PV，省掉原S/P scale分支。会增加BF16舍入，不能声称与原INT8数值逐位一致。原BF16和原INT8控制仍调用原8deb...kernel。

3场景、6阶段的量化FP32参考及CPU抽点、动态Q/K/V/scale且先清空output的Graph、output和segment尾部检查全部通过。另保留相对原BF16输入的近似误差，不替代Math。全部为预先手工量化fixture，计时仅attention+reducer，未包含真实writer/量化成本或整模型。

|场景|BF16 ms|原INT8 packed ms|load/w1 ms|相对BF16%|load/w2 ms|相对BF16%|
|---|---:|---:|---:|---:|---:|---:|
|B8/33|0.015891346|0.016996154|0.018230385|+14.718943|0.019582500|+23.227442|
|B64/4096|0.850252533|0.908722496|1.075207520|+26.457432|1.031527519|+21.320135|
|B64/5120|1.058418083|1.131722164|1.338069725|+26.421661|1.278167152|+20.762029|

B64/4K inline132/load/w1还慢33.313918%。B64控制n_regs85、原INT8为164、load/w1为199、load/w2为192，均0spill；没有硬件counter，资源数只作记录，不单独推断瓶颈。该安排未改善速度，不采用，不进真实writer/服务/Math。

本机复核64份冻结源码/快照、生成源SHA与可逆性、105组原始event、3案例/6阶段检查、stride/grid/量化mode/tile/warp记录。Tensor完整值未保存，不把本机记录审核写成重算GPU结果。生成SHA9a86ce7e6ae60fd4f4f3bd325f45ceb56d5e8e1b2efa3e72e0aac6869e810f10。

原件/workspace/benchmark-results/20261016_kv_int8_dequant_attention_tianshu，4242622bytes，SHAf7b26300c492714096b5ec4eacd3523f1785daa4ccbb9453949edd3a60fa8d99。`python3 experiments/tianshu_kv_int8_20261016/verify_downloaded.py --root <下载目录>`。正式最佳4093.81/Math102保持，未运行新正式/16K。
