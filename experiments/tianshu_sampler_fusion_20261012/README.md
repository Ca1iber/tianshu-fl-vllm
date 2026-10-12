# 天数采样融合实验（105项CPU/FlagGems/PyTorch检查通过）

修正后的独立探针完成105/105检查。初次参考先通过GPU转换FP64导致错误，保留旧21/105的结果，不作为内核否决依据。诊断值CPU转换[1,2,-3]，GPU转换后回CPU[0,0,0]；已改为先传CPU再转FP64。block2048的B1/8/16/64耗时约0.00629/0.01425/0.02390/0.10777ms，相比实际FlagGems0.08119/0.09602/0.14856/0.40487ms。但这是不含RNG/top-p的算子计时，服务效果和新独立数学未验证，未接入生产。

假设：对相同正指数噪声q，实数下argmax(softmax(logits)/q)等价于argmax(logits-log(q))。分块计算logits-log(q)再归约可能省去完整softmax输出及除法中间张量。FP32舍入会改变临界边界，不能声称任意输入bitwise等价。噪声生成、top-k/top-p掩码及采样参数不在此算子探针中修改；尚未接入服务。

计划7种批次1/2/7/8/16/33/64、词表130560，normal/peaked/uniform/masked/all_masked共35输入，每种block1024/2048/4096，共105输出检查。保留CPU FP64独立参考、普通PyTorch参考和服务实际FlagGems softmax/div/argmax参考，失败也保留。固定噪声的三条路径预热后捕获Graph，七轮轮转回放取中位数；耗时不含指数噪声生成和top-p掩码，不能代替完整采样或服务收益。

数值与计时探针通过才设计对应服务A/B；有明显服务收益且独立官方Level3准确率>95%之后才考虑正式4K。专用GPU任务串行、单探针最多30分钟且受11:58租期截止约束，不改site-packages。实际运行目录为`/workspace/benchmark-results/20261012_next_diagnostics_tianshu/sampler`，原始结果、源码SHA及实际FlagGems源SHA用于追溯。

## 服务对照（待运行）

保持 Hybrid/adaptive/FA2 tile128/budget2048，按 control→candidate 串行测试 C64/N192、4K/128，三轮排除首轮。仅替换采样中的 softmax/div/argmax 为固定噪声的指数竞争融合；保留温度、top-k/top-p、全局及每请求 RNG、处理后 logits/logprobs。105 组独立 CPU 对照已通过，但不声称所有浮点输入逐位相同。候选作为进程级钩子，不改安装源码；实际模型必须生成采样路由记录。

在对应独立 Math-500 Level 3 满足严格 >95% 且标准负载确认收益前，不采用为正式最佳。当前 M8 GEMM 对照完成后自动运行，GPU 串行并受 11:58 截止保护。
