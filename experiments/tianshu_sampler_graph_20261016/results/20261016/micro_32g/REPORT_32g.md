# 完整Sampler Graph：V1失败，V2正确但无收益

框架探索独立于正式新最佳4093.81运行。仅C64/BF16[64,130560]、纯TopP、无额外约束时捕获原Sampler；复制动态logits/temperature/top_p到自有缓冲，输出单独clone。保留原BF16→FP32、TopP、softmax、指数采样与int32结果，不提前采样。

V1把原指数采样也捕获进Graph。预热/捕获后RNG状态恢复检查通过，但首组normal、temperature.7/top_p.8/seed111的64token与eager不同，尽管最终RNG状态相同。立即停止该版，不放宽断言、不进入计时或服务。原代码与失败状态保持。实际FlagGems RNG辅助函数在Python取得seed/offset并作为标量传给kernel，这种捕获没有满足此次同序列要求；仅看最终state不足以证明回放正确。

V2只把原指数生成留在Graph外，每次实际抽取一份新FP32 noise，然后Graph使用它执行其余原始算法。9组变输入/温度.7、1、1.4/p.8、.95、.99及normal、peaked、masked，共576token，逐token与eager相同且默认RNG状态相同，BF16输入未修改。另8步eager/Graph交替的512token与全eager序列完全相同，最终state相等，8份旧输出独立且未被后续回放覆盖。12个已记录的不支持配置均被guard拒绝。本微测的processor列表为空，不宣称已验证真实服务活跃processor的回退执行。

7次交替顺序、每次20调用，完整调用包含动态输入copy、原指数生成、Graph replay和独立输出：eager host wall中位3.491333709ms，Graph3.644673410ms，耗时增加4.392009%；GPU流区间3.333820724→3.616483307ms，也无收益。这里只能说明当前完整staged方案无收益，未隔离各copy或Graph引擎的单独成本，不能断言具体瓶颈。

因此不接入vLLM服务、不跑该方向新Math/正式，也不改变当前4093.81最佳。新的框架实验必须解决这次实际完整调用成本，不能只宣传省下若干Python launch。

两版全部原件分别下载至 `service-results/20261016_tianshu/sampler_graph_v1_failed/20261016_sampler_graph_micro_tianshu` 和 `sampler_graph_v2_micro/20261016_sampler_graph_v2_micro_tianshu`。首版4084555bytes/16parts，SHA `226c5d2156eb618a64fa6f3543da82f735f2fcf31a162124e477045eaac7f4da`；第二版4091712bytes/16parts，SHA `d7ffaf66fb8a627f196344205f6b364def3e25a63fc3f20ad28c49ec939cb806`。分片/整包、各63/64份源快照及当前仓库源、实际生成geo源码、全部token与state记录/12guard/14计时的中位数均由本机verifier复核，首版failed状态没有改成complete。

```bash
python3 experiments/tianshu_sampler_graph_20261016/verify_downloaded.py \
  --v1-root ../service-results/20261016_tianshu/sampler_graph_v1_failed/20261016_sampler_graph_micro_tianshu \
  --v2-root ../service-results/20261016_tianshu/sampler_graph_v2_micro/20261016_sampler_graph_v2_micro_tianshu
```
