# Binary32中间表示：数值逐位一致，速度负收益

BI-V150 32G，启动源593c63677a4583c9bd9c081ce6d86f6d62ae0ea8。保持软件8位KV编码/scale和writer，唯一reader变化为构造binary32位模式再转BF16，替代原uint32→uint16→BF16 bitcast。GPU穷举65536编码、256解码码值（含符号零）通过；21场景×两次变数据同Graph，42相位的两byte输出全部逐位相同。167源码快照/生成源、1086数值记录、32 dense检查（69531648元素）、441原始事件组已独立本地核验。未保存完整GPU Tensor，检查范围不冒充全模型质量证明。

|场景|现用byte ms|binary32 ms|耗时变化|
|---|---:|---:|---:|
|mixed64|2.691079|2.709295|+0.677%|
|decode64|0.820343|0.838815|+2.252%|
|decode64_5k|1.016284|1.039204|+2.255%|
|decode256|3.245125|3.331768|+2.670%|
|bounded_fallback|0.147569|0.184116|+24.766%|
|decode16|0.220676|0.225259|+2.077%|
|decode24|0.331412|0.337984|+1.983%|
|decode32|0.419308|0.428070|+2.090%|

B64寄存器从245降到230、B256从245降到229，两者spill均为0；寄存器减少没有带来提速。B64/4K反而慢2.252%、5K慢2.255%，mixed64慢0.677%，容量fallback慢24.766%。prefill staging仍调用相同BF16 attention，约0.04%差异视为噪声。没有足够收益，不接入真实服务、不重跑Math/正式；新best4259.87及其102/105不变。不能把本次寄存器结果外推成具体硬件指令结论。

原件`service-results/20261019_tianshu/soft8_f32_cast_component/20261019_soft8_f32_cast_component_tianshu`，SHA256 `9f69f0550c731c052f0c9160adab1ad9495b92a667247e0457c0f7efb633d9d1`，21分片与整包已验证。

```bash
python3 experiments/tianshu_soft8_f32_cast_20261019/verify_downloaded.py --root ../service-results/20261019_tianshu/soft8_f32_cast_component/20261019_soft8_f32_cast_component_tianshu/micro
```
