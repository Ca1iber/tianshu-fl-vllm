# BI-V150 32G：K8V16 全新 Math105 验证

本次独立运行 **100/105 = 95.238095%**，满足用户提供的官方“大于95%”要求。
105个请求全部成功，0错误、0缺失；没有复用旧双8位方案的102/105，也没有排除错误题。
旧双8位方案上一轮为102/105；两次是不同配置下 temperature=1 的独立生成，
本次结果不用于推断两题差异的因果来源。最终采用仍须同配置完整官方4K通过。

采用官方原始 serve argv，端口9031、模型 MiniCPM5-2B、FULL_DECODE_ONLY、
gpu-memory-utilization=0.85、max-model-len=131072。
EvalScope 使用 math_500 Level3 全105题、batch8、timeout3600、
temperature=1.0、top_p=0.95、max_tokens=32768；没有改生成参数。

错误题索引为18、47、75、87、101。104题正常 stop；101题触及32768 token上限，
仍按真实评分计为错误。最后一题完整等待，没有提前截断或重抽样。

实际 API 的41项环境与本次196个冻结源码、全部生成源和各105条 prediction/review
记录已在下载后独立核算。原2048调度预算、max_num_seqs256、async scheduling、
75点预热和严格TopP32K采样保持原配置；K为软件8位，V为BF16，12KiB物理页。
真实cache spec/view、BF16 V零拷贝别名及512MiB共享arena的显存预算通过运行时审核，
before/after metrics 均零抢占。Math并发8不替代正式C64的绑定验证。

启动源码0e8d9b8a1256951cbb689f8b21f9a858454b2a1b；运行时只新增报告、复现脚本和
独立后续实验准备文件，196个目标源码不变，归档时Git HEAD7353617。
根目录 `/workspace/benchmark-results/20261019_k8v16_engine_math_tianshu`。
完整归档6505917字节、25原分片，SHA256
`5402b9ec31e082c2eae23b30adcd81aa33eebe3cfbff42a6081577b1eefb2497`。
本地 `service-results/20261019_tianshu/k8v16_engine_math/20261019_k8v16_engine_math_tianshu`，
`verify_validation_downloaded.py --root <此目录>` 独立核验通过。

随后同196个源码和41项环境启动完整原始4K/1K、C64、N256四轮测试，首轮排除；
正式结果另报。本报告不把先前N64筛选的4886.48 total tok/s当作正式分数。
