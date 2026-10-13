# 天数 BI-V150 32g：BF16贪心采样完整路径局部实验

将BF16 logits直接加载为FP32归约，避免整张logits FP32临时张量以及公开int64→int32转换。公开SamplerOutput保持int32[B,1]。仅支持B1..64/V130560、连续词表维、纯贪心、无logprobs/惩罚/白名单/禁词/非argmax不变processor/活跃thinking-budget；其余回退原Sampler.forward。BF16到FP32转换精确，归约不改变精度或随机采样。

48场景覆盖4个shape/stride组合×BF16/FP32×6模式；CPU参考、Graph、输入位与RNG不变全部通过。40个非NaN场景与显式FlagGems环境下原Sampler逐位一致；8个NaN场景原FlagGems首NaN处理反例保留，候选仍与CPU首NaN参考一致。9个metadata拒绝条件和CPU/词表边界已检查；尚未把该静态guard检查当真实服务命中证明。

|形状|类型|原Sampler ms|候选 ms|耗时变化|
|---|---|---:|---:|---:|
|B1/strided=False|torch.bfloat16|0.057233|0.012132|-78.80%|
|B1/strided=False|torch.float32|0.054510|0.012070|-77.86%|
|B7/strided=False|torch.bfloat16|0.064640|0.013474|-79.16%|
|B7/strided=False|torch.float32|0.055185|0.014314|-74.06%|
|B64/strided=False|torch.bfloat16|0.172253|0.046274|-73.14%|
|B64/strided=False|torch.float32|0.088590|0.063163|-28.70%|
|B64/strided=True|torch.bfloat16|0.370787|0.045652|-87.69%|
|B64/strided=True|torch.float32|0.368316|0.074334|-79.82%|

每项7轮交替顺序×50次Graph replay，原件保留每轮时间；基线明确为flag_gems.use_gems内完整原Sampler.forward。局部收益不当作服务吞吐，成对4K短测另行进行，新Math/正式benchmark未运行；当前正式最好3640.16保持。硬件profile：TODO有效图内kernel覆盖。

原包2574188bytes/10分片，SHA256 524aded80e659bfb2a077459e91ce3c349d2a6077a3fbdf343ca3d3dc5f3d916，分片/整包/三份源已本地核验。原件service-results/20261014_tianshu/greedy_bf16_micro/20261014_greedy_bf16_micro_tianshu。
