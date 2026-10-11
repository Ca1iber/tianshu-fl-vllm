# 天数 Prefill 分桶和启动预热实验

针对用户在沐曦发现的动态稀有 GEMM 尺寸运行期调优，在天数插件独立实现，不修改沐曦仓库或官方 benchmark。分支 exp/tianshu-prefill-buckets-20261011，基点 e030a9d。

## 机制

新开关 VLLM_FL_ILUVATAR_PREFILL_BUCKETS 默认0。显式开启时，仅适配本次 Iluvatar、MiniCPM Llama42层/hidden2048/intermediate6144/GQA16:2、未量化、单卡、无LoRA/投机/多模态、budget<=2048及FULL_DECODE_ONLY。

75点：[1,2,4]，8..256步长8，272..512步长16，544..1024步长32，1152..2048步长128。565补到576，1537补到1664。最大点受现有token预算限制，不增加scheduler预算或模型最大长度。

天数保留官方FULL_DECODE_ONLY。配置列表扩到75点，但实际Decode图仍只捕获请求数范围内的点；Prefill/混合批次的模型GEMM向上补齐到这些行数，以普通编译路径执行，不宣称重放大Prefill Graph。启动时显式_dummy_run预热全部75点，包括Decode图未覆盖的高M点。编译范围保持原动态范围，不生成75份静态模型。

ModelRunner仅在非uniform Decode、Graph NONE、非强制eager、无cascade/encoder时补齐。现有attention metadata仍使用真实token数；位置padding归零；无效KV slots沿用-1；logits indices选择真实请求的最后token。模型、Attention、KV dtype、Decode16段策略和采样保持原版。上一次三个Attention候选开关全部0。

## 验证设计

CPU检查覆盖75点数量、固定例子、每个token数的预算边界及关闭/不支持模式回退。实际GPU服务先原版后候选，串行运行，不删除或人工更换已有调优缓存。

稀有请求：565/577/599/1537/1561输入、16输出、固定种子、temperature0；比较A/B文本及usage。随后5轮短4K：每轮16请求、并发16、4096输入/64输出，固定成对种子、temperature1/top_p.95、ignore_eos，轮间前缀不同以避免prefix cache复用。该负载不等于官方4096/1024、256请求四轮流程，全部结果标为实验。

TRITON_PRINT_AUTOTUNING=1记录首次配置选择。分别保存启动期和服务就绪后的mm_kernel与其他算子记录，不把所有停顿都归为GEMM。该日志中的bench_time未必覆盖完整JIT等待，服务wall time另测。每条响应核对输入/输出token数，五轮计算后四轮最大差值/均值。

原版短对照已完成：首轮53.419秒，后四轮20.253/20.254/20.277/20.276秒，后四轮波动0.1209%；运行期24条mm_kernel调优以及2条索引调优。未据此归因全部首轮停顿。

候选安装只使用pip --no-build-isolation --no-deps，保持底层Torch/Triton/vLLM版本。源码安装哈希与所有日志在工作区service-results/20261011_1910_prefill_buckets_tianshu。MATH-500尚未重测，不能沿用原版96.19%作为候选成绩。

## 执行

```bash
python3 run_bucket_probe.py --output /workspace/benchmark-results/probe/control --buckets 0
# 原版完成并停止后，再安装候选插件源码。
VLLM_VENDOR= python3 -m pip install --no-build-isolation --no-deps /workspace/vllm-plugin-FL
python3 run_bucket_probe.py --output /workspace/benchmark-results/probe/candidate --buckets 1
```

所有服务由脚本独立进程组持有，结束或信号只清理自己的进程；23:58截止有watchdog。新开关默认关闭，可保留候选源码并返回原计算路径。

## 第二个问题：最后token hidden-state索引

仅75点候选的运行期mm_kernel新增调优为0，但首轮短4K耗时66.115秒；后四轮20.269/20.266/20.284/20.305秒，波动0.1928%。首轮仍有通用_index_linearized索引调优。因此未认定整体稳定性成功。

加入第二个默认关闭的VLLM_FL_ILUVATAR_LOGITS_GATHER=1，在已启用分桶且满足同一模型/设备条件时，将vLLM采样前从hidden_states选取最后token行改为固定launch的拷贝。源行数、输出行数与stride不作为JIT特化键，无autotune，不按输入M生成新代码；仅支持调度器生成的有效非负行索引。Dummy预热和真实采样使用同一路径，并分别预热int64/int32索引。未改变logits、采样算法、模型计算或KV。

独立CPU索引位级检查120组通过：5种源行数（565/576/1537/1664/2048），连续/strided列，int32/int64索引，0/1/3/16/64/256输出行，随机顺序与末行索引。改变源值和索引后3次CUDA Graph重放仍位级一致。仅分桶5组实际模型greedy请求的16输出token文本及usage与原版相同；不能替代MATH-500。

联合候选另用 --buckets 1 --gather 1 运行同样5轮短对照，完整前一版失败记录保留在candidate目录；联合记录保存在gather_candidate目录。

## 最终结论

联合版五轮短4K耗时43.739/20.306/20.274/20.239/20.304秒，运行期GEMM和索引调优均0，后四轮波动0.3308%。消除了调优但没有完全消除首轮开销，不认定达到0.1%稳定性，默认仍关闭。所有版本每轮16/16成功，input65536/output1024；五组greedy文本及usage全部相同。详见comparison.json和工作区同名报告。未重测MATH-500，未运行完整4K/16K。
