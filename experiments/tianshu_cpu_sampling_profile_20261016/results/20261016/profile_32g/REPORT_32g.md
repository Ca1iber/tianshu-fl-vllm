# 天数框架CPU调用诊断：BF16 logits与中立采样约束已确认

本次单独诊断完整通过：两组64并发、4096输入/256输出，全部128响应的正文、usage、输入种子与时间核验；seed777前后canary正文相同。第二组连续纯Decode32步后，采集16个实际FULL/C64 frame，每帧分别cProfile execute_model与sample_tokens。没有IXsys或GPU events，不是吞吐测试。

本轮使用原3992.97版nativeTopP/CV/KV16-w1/S8/75warm及官方原始serve argv，35基础环境加4诊断项。GEMM+几何TopP新版正式成绩另记4093.81，本诊断未假称运行新版算子。源码6bac196，运行和下载期间146份运行源与快照SHA保持；与同期正式/Math源去重170份也在报告同步前后核验，用户未提交改动原样保留。

侵入式cProfile的host wall中位数：execute_model7.383968ms、sample_tokens3.438765ms；线程CPU分别7.320253ms、3.439460ms。11008条原始函数记录及父调用均重新解析。execute_model的_prepare_inputs平均累计约5.321ms，包含内部调用；sample_tokens中FlagGems run每帧7次、平均累计约0.703ms，另有多个to/copy和日志记录。累计父子调用重叠，不能相加，也不能将这些数值当作正常运行host耗时、GPU kernel时间或其占比。

16帧实际进入采样器的logits为BF16[64,130560]、连续，sample_hidden_states为BF16[64,2048]；logits、hidden、sampleHidden、temperature及top_p各自在这个窗口地址保持相同，top_k=None。地址稳定结论仅覆盖本窗口，不能依赖它缓存任意后续请求。采样Graph候选应复制动态输入/temperature/top_p到自己拥有的静态缓冲，并返回独立输出，避免偷用旧地址和异步覆盖。

全部16帧all_random=True、all_greedy=False、no_penalties=True，无per-request generator、badwords、allowed-token mask、logprobs或thinking holder。内置processor列表并不为空，但MinP.min_p_count=0、MinTokens.min_toks与LogitBias.biases为空，因此此次约束中立。后续Graph必须逐次检查这些条件，未知processor或活跃约束回退原路径，不能仅凭列表非空/为空猜测。

下一实验拟在原sample调用阶段捕获完整Sampler，不提前到模型forward，保留原BF16→FP32、TopP、softmax、指数采样和默认RNG。先独立验证变输入、变temperature/p、默认RNG状态与混合eager/Graph序列、旧输出生命周期和回退边界；只有实际通过后才做服务4K对照，目前没有实现或收益结论。

原件 `service-results/20261016_tianshu/cpu_sampling_profile/20261016_cpu_sampling_profile_tianshu`，6994775bytes/27分片，SHA256 `2efdb3c1a0652991cdd618569c8d90723f586c225b054228614014baf129362b`。所有分片、整包、146份源快照、38基础路由字段和实际39环境、128响应/两canary、16帧/32函数profile、指针与processor信息均通过本机独立verifier。报告旁保留完整机器审计，不从侵入式profiling推算模型吞吐收益。

```bash
python3 experiments/tianshu_cpu_sampling_profile_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/cpu_sampling_profile/20261016_cpu_sampling_profile_tianshu
```
