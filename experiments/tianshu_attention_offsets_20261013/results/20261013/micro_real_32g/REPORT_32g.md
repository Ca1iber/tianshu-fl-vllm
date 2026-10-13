# 天数 BI-V150 32g：真实Decode热路径的32位元素偏移

按用户继续int64→int32要求，审计到native attention地址步长注解与显式元素offset casts仍int64。复制冻结源bab2c0b9...，仅将整数地址步长与element-offset算术窄为int32，浮点主体、物理指针及KV storage dtype不变，不改installed source。

每个参与tensor检查max_element_offset=sum((dim-1)*abs(stride))+2048*sum(abs(stride))，包含保守masked-tile余量，必须0≤offset<2**31；不满足则服务候选回退original。MiniCPM geometry Q16/KV2/D128/BF16，无quant/bias/sinks，只3D Decode，不影响已验证L0 Prefill。控制与候选均TILE16/w4/stage1、同adaptive分段。

第一轮8布局均与GPU FP32参考通过且候选逐位等原64；随后真实共享交错KV缓存torch.stack((K,V),dim1).unbind1，Qstride[2560,128,1]、K/Vstride[8192,256,128,1]，10场景同样全部通过且bitwise equal。七轮旋转CUDAgraph计时、中位数，float运算未改变。

|batch|KV最长|segments|原64/ms|地址32/ms|速度提升|逐位一致|
|---:|---:|---:|---:|---:|---:|---|
|1|5120|64|0.038571|0.038822|-0.645942%|是|
|4|5120|16|0.130326|0.107303|21.455939%|是|
|8|5120|16|0.226954|0.203266|11.653577%|是|
|16|5120|4|0.466865|0.376505|23.999683%|是|
|24|5120|4|0.746638|0.705085|5.893312%|是|
|30|17408|4|2.815043|2.461231|14.375370%|是|
|32|17408|4|2.842239|2.517610|12.894310%|是|
|48|5120|4|1.174114|1.127700|4.115850%|是|
|64|5120|4|1.488123|1.493199|-0.339879%|是|
|8|32768|16|1.350750|1.209427|11.685105%|是|

结论：中等批次有明确attention micro收益，B16约24%、B30/17408约14.375%、B32/17408约12.894%；B1/B64没有收益。服务原型仅4≤batch≤48及strict offsets安全时选择32，B1/B64等原版回退。并非把所有数据/物理地址变32，不强改公共slotmapping。

真实16K输入/1K输出/C64/N64双组3轮（排首）服务实验已启动，freshcontrol0/candidate1；当前只有micro收益，候选独立Math/新正式成绩未跑，未采用。若实际收益明确再质量和完整benchmark。正式最佳仍4K3222.52/16K2440.62/Math102/105。

真实布局原始2033328bytes/8parts，SHA256 43a42db5358e5553e0214017c3faca6e95ee58f2fd27343282a45e21673338e4；初8布局2032221bytes/8parts，SHA256 a7f0298303e63a13e3beb47b6c5ca3a2e8c4d5dcaeec8e5572ae309290548cfa，均全部本地核验。真实helper/probe与source snapshot逐字相同。未采集硬件profiler。
