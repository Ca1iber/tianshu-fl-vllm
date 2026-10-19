# Top-P插值：数值通过，BF16负载没有明确提速

2026-10-19，BI-V150，V2源码96ccba9。保持FP32概率、TopK原生路径、32次搜索上限、1e-12终止和CPU1e-6质量门槛。V1极小1e-38常量导致候选隐式FP64类型编译冲突，27个非TopK候选失败；完整失败与控制数据保留。独立V2以正常FP32范围的1e-30仅调整插值探测点下限，所有10场景/7配置原始与改变输入Graph数值通过。实际生成TTIR没有f64类型。

|B64分布|当前32K GEO ms|Secant ms|Illinois ms|
|---|---:|---:|---:|
|normal|2.593457|2.312338|2.410489|
|peaked|1.673017|1.714005|1.640563|
|bf16_normal|2.463197|2.438888|2.459949|
|bf16_peaked|1.899803|1.930742|1.937072|

普通FP32的改善不能外推实际BF16 logits。BF16 normal几乎持平，BF16 peaked略慢，暂不做服务或新Math/正式benchmark。当前最好4093.81 tok/s与Math102/105不变。

本地重算4928条逐行质量记录、490条原始event、全部12份冻结源码及生成源码/TopK AST通过。GPU完整Tensor没有序列化，离线核验限于记录与来源；此探针未执行采样RNG或模型服务。

完整V2包5812568字节，SHA256 `a548714f847cae6a18be2df091fb4318e1f94e1ac0ec6b138b70d0ec3bd844b7`。原件位于service-results/20261019_tianshu/topp_interpolation_v2_micro/20261019_topp_interpolation_v2_micro_tianshu。V1原件保留在topp_interpolation_micro，整包SHA256 `c407f36e649670d2e001518c9403fc6e7119b8c9ca7cb9c741b7f8c17b371b18`。
