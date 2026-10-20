# 天数 BI-V150：原生 Decode + FA2 Prefill 优化报告

这轮将 vLLM 的 Prefill 和 Decode 分开处理：纯 Decode 沿用原生3D Attention，Prefill使用FlagGems FA2，混合批次通过原生请求重排拆成两条路径。短4K/1K实验吞吐提升22.41%，TTFT降低41.73%；完整官方4K复测已完成：2286.09 total tok/s、TTFT7569.57ms；相比本机原版吞吐+21.82%、TTFT降低38.54%。独立MATH-500 Level3为100/105（95.24%），原版101/105（96.19%）；用户明确官方门槛为正确率大于95%，候选已达标；不再以追平原版作为采用条件。四轮1024个计分请求全部成功；天数推荐启动配置开启候选路径。

## 改动与依据

原生3D Decode的短扫覆盖1/8/16/32/64路、4096/5120 KV、ragged及strided布局，40组固定FP32检查通过。调整分段有局部收益；更大tile出现2组失败，保留原始记录，未启用这些配置。实际profile显示Prefill与混合路径值得优先处理；FULL_DECODE_ONLY的Torch profiler没有记录Graph内Decode，不能用其GPU总和代替整条服务时长。

插件新增VLLM_FL_ILUVATAR_HYBRID_ATTENTION，默认0。原生metadata builder以query长度1作为重排界线，每步CPU检查一次边界；混合批次的Prefill GPU cu_seqlens只重基一次，全层共享。原生KV写入、纯Decode、采样及vLLM核心安装文件保持原路径。服务显式使用FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1。未使用重排权重、75点padding或logits gather。

限定当前单卡BF16 MiniCPM5-2B：42层、hidden2048、16 Q heads/2 KV heads、head_dim128，普通causal attention。不适用的mask、量化、多模态、LoRA、投机或DCP配置保留原生回退。

## 数值与完整模型质量

经真实native builder/forward wrapper验证纯Prefill、chunk后缀、Decode1/64、mixed64、ragged mixed、未重排回退、输出padding和CUDA Graph。候选通过独立FP32检查（固定atol=.02、rtol=.01）；纯Decode和回退与原版位级一致。原版一个Prefill场景超过固定容差，失败记录保留；更接近张量参考不代表模型准确率持平。

| MATH-500 Level3 | 正确/总数 | 准确率 | 请求错误 |
|---|---:|---:|---:|
| 原版独立评测 |101/105|96.19%|0|
| 新候选独立评测 |100/105|95.24%|0|

候选105条预测、评分、执行统计均完整。官方temperature1/top_p.95/max_tokens32768、batch8参数不变；两题达到token上限，其中一题评分仍正确。逐question_id核对：原版正确→候选错误2题，反向1题，共同错3题。没有单独控制成对采样seed，不能把差距直接归因为随机波动或计算路径。未修改官方评分，也未复用原版准确率。

## 短服务实验

两组采用相同serve参数、同客户端与配对提示词seed，每轮更换输入前缀以避免跨轮完整prefix cache复用。4096输入/128输出、16路/16请求、五轮跳首轮：3024.12→4547.85 total tok/s（+50.39%），TTFT10251.90→6089.29ms。

进一步测试4096输入/1024输出、64路/64请求、三轮跳首轮：

| 版本 | 后两轮 total tok/s | output tok/s | mean TTFT ms | 最大差/均值 |
|---|---:|---:|---:|---:|
| 原版 |1902.54|380.51|39749.66|2.2663%|
| 混合候选 |2328.85|465.77|23163.60|0.4945%|

后两轮吞吐+22.4076%，TTFT降低41.7263%。两组各199请求（含7个短诊断），398请求全部成功。首轮较慢仍保留，不计均值；只有两轮统计，不宣称0.1%稳定性。64请求只有一波，装载与排空占比高，不能与官方256请求均值直接比较。客户端显式temperature1/top_p.95；本机官方客户端不指定时使用模型同样的默认参数，相关源码核对已记录。

## 官方4K复测

用户已明确授权短测明显提速后完整复测。本次使用原样官方benchmark脚本：[[4096,1024,64,256]]，四轮跳首轮，不启动16K。脚本SHA256为0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565。服务保持port9031、FULL_DECODE_ONLY、gpu-memory-utilization0.85、max-model-len131072。

| 官方4K | total tok/s | mean TTFT ms |
|---|---:|---:|
| 本机原版完整均值 |1876.67|12315.86|
| 用户提供公开基线 |2028.01|11573.53|
| 新候选完整均值 |2286.09|7569.57|

## 版本、环境与数据

插件分支exp/tianshu-native-attention-20261011，最佳检查点8daa9520a38295957d4968a529b808350a49b313（tag tianshu-best-hybrid-20261012）；生产helper引入于25496a5，SHA256为5c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4。算子库58c31398b5ffaa9baac760f07a56ec41ffe0218a，本轮没有新增GPU kernel源码修改。

实际可见BI-V150 32GiB；Torch2.10.0+corex4.5.0、vLLM0.24.0+empty、FlagTree0.6.2a3+iluvatar3.6、EvalScope1.11.1。插件非editable安装，package metadata仍是de5b668构建标记，实际工作源码和运行源码按SHA追踪；不能仅凭包版本推断改动。实际Flash导出为flag_gems.ops.attention.flash_attn_varlen_func，安装路径与attention/flash_api/flash_kernel源指纹及快照均保存。

材料位于本目录下20261011_2100_native_attention_tianshu（数值、profile、MATH、16路短A/B）、20261011_2238_hybrid_4k1k_c64_tianshu（64路短A/B）及20261011_hybrid_official4k_tianshu（正式四轮）。原始JSON/JSONL、CSV、日志、源码指纹、错误记录和分块校验表保留。本轮没有修改或推送沐曦仓库；GPU实验串行，只清理自有进程。用户已续约至2026-10-12中午12:00，停止截止11:58。

当前采用结论：候选100/105已通过官方大于95%的正确率门槛。正式吞吐复测已确认收益；天数专用启动配置serve_tianshu_minicpm.sh显式开启新路径，作为当前推荐配置。原版准确率对照仍保留，不作为否决条件。

官方准确率门槛及用户澄清记录见各实验目录accuracy_requirement.json。源码通用opt-in开关与天数服务启动配置区分：新天数启动配置开启HYBRID_ATTENTION=1、COMPACT_VARLEN=1，与独立数学评测和正式性能复测一致。

正式四轮全部完成：首轮1892.68/10479.00ms跳过；第2–4轮严格按原脚本求均值。正式成绩2286.09 total tok/s、457.22 output tok/s、TTFT7569.57ms，四轮1024/1024成功；原始CSV与汇总JSON见正式目录。

正式均值已独立审计，采用当前天数优化配置。相对公开基线吞吐+12.7258%、TTFT降低34.5958%。三轮吞吐最大差/均值5.6616%，仍有稳定性优化空间，不宣称0.1%。
