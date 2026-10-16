# 天数：GEMM与TopP32K实际最大值组合，4K短测提升1.68%

运行源码dbb06c2c8fac8191ec8ebb61bd1c8b88f2adb0fd，后续563f9b0仅新增未启动验证脚本，55份冻结运行源码全部不变。官方原始serve argv，实际默认2048 token预算/256序列/async=True/75预热点、CV/KV16/S8、nativePrefill L0、原采样器。未改变调度参数、精度、RNG或输入长度。

候选组合：M2048的O/down/gate_up三项已验证GEMM配置，以及严格概率域TopP32K/16Ktrunc/w16，以actual global max而非sampled avg+10std作为exp归一化偏移，普通FP32归约。B8..64普通TopP生效，TopK和其他路径回退。原生TopP严格语义误差保留为对照，候选10输入严格CPU FP64语义/Graph检查已独立通过；整轮4224份逐行记录（含原生对照的严格语义错误）均已重新核算；本次服务仍不能替代Math正确率。

|模式|轮|total tok/s|output tok/s|TTFT ms|运行时autotune记录|
|---|---:|---:|---:|---:|---:|
|control|1（排除）|3752.194841|750.438968|16801.077402|0|
|control|2|3929.146300|785.829260|13145.608720|0|
|control|3|3943.091966|788.618393|13202.255511|0|
|candidate|1（排除）|3359.684542|671.936908|26824.908380|0|
|candidate|2|3991.354770|798.270954|12761.488111|0|
|candidate|3|4013.441467|802.688293|12778.571716|0|

每模式N64/C64、4K输入/1K输出、3轮去首轮，每轮不同输入seed、两模式配对。后两轮均值3936.119133→4002.398118 total tok/s，+1.683866%；TTFT 13173.932116→12770.029914ms（-3.065920%）。候选首轮明显慢于控制，原始值保留，原因未单独隔离；不将排除首轮等同无启动开销。

控制/候选稳态轮间差距分别0.354300%/0.551837%。提升未达预设2%扩大验证门槛，未启动准备好的新Math或正式4K，不采纳为新正式最佳。N64短测不能直接与N256正式均值比较。正式最佳仍3992.97 total tok/s、798.60 output tok/s、TTFT4443.93ms，Math103/105。

384请求全部成功；文本、usage、输入seed、用时、4个C1确定性canary、16份metrics零preemption、实际38环境与原始argv、55冻结源码/当前源码、候选实际B64 TopP source af8ffb90及3项M2048配置均下载复核。C1 canary走回退路径，不作为B64正确率证据。0条打印的运行期调优记录，不声称证明无任何未打印JIT工作。

原件 ../service-results/20261016_tianshu/gemm_topp_max_service/20261016_gemm_topp_max_service_tianshu，6046319bytes/24parts，SHA256 `ab5fb998780e0618e59186999d8dc6098e59a62426c2f7aeb8c6724833059140`。所有分片和整包校验通过。runner41303已终态，两个API/Engine进程均退出，GPU无运行进程。

```bash
python3 experiments/tianshu_gemm_topp_max_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/gemm_topp_max_service/20261016_gemm_topp_max_service_tianshu
```
