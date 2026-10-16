# 天数NHD/HND KV布局：组件基本持平，不采用

保持BF16、page16、逻辑[num_blocks,2,16,2,128]及真实builder/KV writer/attention；物理布局NHD与HND仅token/head次序不同。该微测手工构造框架原生stride接口对应的view，尚未改变完整模型allocator。两边使用当前GEO最佳38项配置中的.cv/L0/KV16-w1/S8 attention。

16场景覆盖prefill/chunk/ragged/mixed/B1至B64 Decode、4K/5K及短边界/非连续页。每例2个阶段就地改变Q/K/V、slots和长度并重放原Graph，FP32与CPU抽点参考、整块物理KV写入、输出/segment尾部都通过；NHD与HND输出逐位相同。完整GPU Tensor未序列化，本地核对源码及记录，不声称本地重新执行了数值检查。

|场景|NHD ms|HND ms|耗时变化%|
|---|---:|---:|---:|
|prefill2048|0.713018656|0.713836145|+0.114652|
|chunk2048|1.841402245|1.844203949|+0.152151|
|mixed64|2.890940475|2.891163254|+0.007706|
|mixed_ragged|1.112735176|1.103875732|-0.796186|
|decode64|0.882426357|0.882406902|-0.002205|
|decode64_5k|1.092245388|1.091549015|-0.063756|
|decode1_33|0.020137499|0.020409615|+1.351290|

B64/4K仅-0.002205%、5K-0.063756%，无明确主负载收益，不采用，不做服务/新Math/正式benchmark。没有从mixed_ragged局部-0.796%外推整体收益。

61份冻结源及当前自有文件SHA、38项当前最佳环境、实际原生prefill/CV/KV16-w1/S8、16场景stride、32阶段检查记录、224组原始event均已本地独立核对；每个event为20次Graph平均。原件/workspace/benchmark-results/20261016_kv_hnd_micro_tianshu，4211100bytes，SHA65701a8d7badf4b63eb706cdf465ca0d4870ef4908afb2b129408a7dbe4df73a。

复核：`python3 experiments/tianshu_kv_hnd_20261016/verify_downloaded.py --root <下载目录>`。正式最佳4093.81 total tok/s、Math102/105保持。无16K、安装源或沐曦工作树改动。
