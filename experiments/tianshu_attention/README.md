# 天数 Attention 第一次优化检查点

仅用于 BI-V150 天数仓库。沐曦库只读参考；不移植 C500 的预排权重与 tile 常量。

## 路径与候选

原版 dispatch 日志显示 default.flagos，但 VLLM_FL_USE_FLAGGEMS_ATTN 默认 0，实际返回 vLLM TritonAttentionBackend。独立 4K profile 中 kernel_unified_attention 占 GPU 时间 76.9%。现有 FlagGems Attention 单请求更快，但直接开启后真实混合批次退化，原始失败记录完整保留。

新的天数候选在 FlagGems 中增加默认关闭的 FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1：以 floor(cu_q[i]/BLOCK_M)+i 构造查询 tile 区间，在 GPU 上查找所属序列。Grid 数量从 ceil(max_query_len/BLOCK_M)*batch*heads 变为 (floor(total_query_tokens/BLOCK_M)+batch)*heads。每条序列有足够 tile，最多多一个；尾部空 tile 返回。无需把 GPU 序列长度复制到 CPU，不增加持久权重或 KV 显存。

仅在 Iluvatar、存在 cu_seqlens_q、无 dropout 且显式开启时选择该路径。单查询 GQA 交换路径仍使用原 grid。修改计算块的索引与数量，Attention 算式和 tile heuristic 保留。

第二个默认关闭的候选 VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN=1 使用 vLLM 标准 reorder_batch_threshold=1，把单 token 请求排到前面。metadata builder 只读取现有 CPU query starts，验证连续 Decode 前缀且其余请求长度均 >1；每步一次构造重基的 Prefill starts。Attention 层把 Decode/Prefill 分为两个 FlashAttention 调用，从而保留 Decode 的 GQA query-group 交换优化。输出写入互不重叠的原缓冲切片，并返回原完整输出缓冲。DCP、多级 prefix cascade、纯 Decode、纯 Prefill 和不满足前缀条件的输入沿用原分支。

候选仅显式开启于 Iluvatar，Decoder forward 才执行分离，未更改 KV 缓存布局、模型/权重、采样或 CUDA Graph 设置。

## 数值验证

validate_paged_attention.py 使用独立 FP32 QK/softmax/PV 与明确的右下对齐 causal mask；固定采用 FlagGems v5.3.5 测试的 atol=0.02、rtol=0.01。新 compact 输出还必须与原 FlagGems FlashAttention 逐元素完全一致。分离 GQA 后的输出对固定 FP32 参考通过相同容差；因计算 tile 路径不同，不宣称分离路径与原混合 kernel 位级一致。额外直接调用生产 metadata builder 和 forward，覆盖分离条件、回退以及输出 padding。原 vLLM 通用 Triton 路径存在的参考差异单独保存，不通过放宽容差消除。

覆盖 2048-token Prefill、带 2048 KV 前缀的 Chunked Prefill、单路和 64 路 Decode、不连续的分页缓存与 QKV 查询切片、混合变长批次，并把长 Prefill 放在 64 路批次的首位/末位。GPU 实验串行运行。

## 运行

先安装天数两个候选源码包，保持底层依赖：

```bash
python3 -m pip install --no-build-isolation --no-deps /workspace/FlagGems
VLLM_VENDOR= python3 -m pip install --no-build-isolation --no-deps /workspace/vllm-plugin-FL
```

验证与一轮 4K 比赛负载：

```bash
export VLLM_PLUGINS=fl
python3 validate_paged_attention.py --output /workspace/benchmark-results/check/validation.json
python3 run_4k_probe.py --output /workspace/benchmark-results/probe4k \
  --attention-backend flaggems --compact-grid --split-mixed \
  --validation /workspace/benchmark-results/check/validation.json
```

run_4k_probe.py 启动官方服务参数，暖机不同随机前缀，运行原官方脚本中相同的内层 vllm bench serve 命令。负载 4096/1024、并发 64、请求 256。严格检查成功请求数和输入/输出 token 数，保留客户端/服务端日志，完成或信号停止后只关闭自己的进程组；遵守本次预约 23:58 的停止时间。

**单轮仅是实验数据，不是四轮去首轮后的官方成绩。算子验证不能替代 MATH-500。旧版 96.19% 只属于原版，不能计为新候选精度。**

正常启动候选可执行 bash serve_candidate.sh；返回原路径需把 VLLM_FL_USE_FLAGGEMS_ATTN、FLAGGEMS_ILUVATAR_COMPACT_VARLEN 和 VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN 置为 0 后重启。

结果与失败记录归档在工作区 service-results 对应日期目录；最终数据另见报告。

## 下一次短实验的重点

原服务 TritonAttentionMetadataBuilder 会在请求数不足launch阈值时使用3D分段softmax与reduce；七组算子测试中的original是未提供分段缓冲的2D调用。真实日志中候选纯Decode比原版慢，Prefill优势可能被抵消。下一次应对真实3D Decode做独立FP32正确性与device event对照，并检查FlagGems Decode的分段/布局方案，避免再次先跑长benchmark。

## 本轮结果：不采用，保留失败检查点

单轮4K（256/256成功，输入1048576、输出262144）：duration763.71秒，total1716.26、output343.25 tok/s，meanTTFT9542.83ms。今日原版官方均值1876.67 tok/s、TTFT12315.86ms；本次观测吞吐−8.55%、TTFT−22.52%。公开基线2028.01 tok/s、TTFT11573.53ms。候选未达到整体性能目的，保持默认关闭，不继续跑四轮。未重跑MATH-500。

单轮统计见single_run_result.json，算子摘要见operator_summary.json。完整日志与成功/失败实验在工作区service-results。所有自有服务与benchmark进程已停止，最终GPU快照无运行进程。没有修改或推送沐曦仓库。
