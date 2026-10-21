# 天数 DSpark 重新接入前的语义核查（CPU/源码，未跑新模型）

此记录不产生新吞吐、接受率或准确率成绩。推荐版本继续为正式 4K 5586.28 tok/s、TTFT 3780.63ms、Math 102/105。当前多查询 attention 组件只是后续投机接入的准备。

## 旧有效评测的接受率

从旧 `prefix2_standard_4k1k` 第三轮前后 Prometheus counter 相减：82,190 次验证、164,380 个提案 token、48,854 个接受 token。平均每次验证接受 0.594403 个草稿 token；第一个位置无条件接受率约 36.93%，第二个约 22.51%。这不是新评测，也不能与当前推荐版吞吐混作同一模型配置。精确计算、全部计数及源文件 SHA 见本目录 JSON/TXT。`1+accepted/drafts` 只代表名义输出长度，结束/EOS/长度截断不保证每次有一个 bonus。

比赛 benchmark 脚本明确调用 `vllm bench serve --dataset-name random --ignore-eos`。本地旧小评测同样使用随机 token 输入，且仅取 2 个提案。模型卡自然文本、7 个提案的接受长度不是同一工作负载，不能仅凭二者差异认定适配有错。

## 已核对的实现

- 本地 ModelRunnerFL 对 dflash_config.target_layer_ids 执行 `i+1`，42 层 Llama 在每层输出后按 `idx+1` 收集含 residual 的辅助隐藏状态，选中 [2,11,21,31,40]，与 DSpark 训练侧按原层号+1取 hidden_states 的规则一致。训练侧定义见 [DeepSpec common.py](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/common.py)。未发现这里漏掉+1。
- 本地 DSpark 继承 DFlash Qwen3 backbone，辅助特征经 FC 变为 H2048，再由 hidden_norm 后投影为五层 context KV。本地没有 projector_type 分支本身不构成错误证据；[当前官方 Qwen3 DSpark](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_dspark.py) 也复用 DFlash backbone。当前源随 main 变动，此处只作机制对照，并非固定版本逐位等价证明。
- 本地 query 槽 0 为 anchor，所有七个训练查询槽参与非因果 backbone，只取所需前缀 hidden 计算提案。Markov bias 从 anchor token 开始，之后逐个使用前一提案，与[官方 DSpark sequential sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py) 的顺序一致。未证明整个模型数值等价。
- 本地目标和草稿 config 都没有额外 logit_scale，默认 1；不能把低接受率归因于已证实的缩放差异。
- 本地输入准备根据最后一个有效 target 位置+1生成 anchor 位置，在 padded rejection 模式下减掉拒绝 token 数再取位置；有效 seq_len 同样扣拒绝数。query 非因果，旧被拒 KV 在 query 写回前可能存在，最终有效范围和写入顺序仍需 GPU 输入/槽位诊断，未在本次源码审阅中宣称端到端验证。

## 已确认的边界不一致及下一步

旧 scheduler hook 已把 lookahead 提高为 7，DSpark backbone 也固定写七个查询槽。但 ModelRunnerFL._input_fits_in_drafter 仍沿用 DFlash `num_spec_tokens+1`。当只提案 2 个时，长度检查按 3，实际写入按 7：模型最大长度末尾四个位置可能被错误放行。这是源码层面确认的保守长度检查缺口，尚未在服务中修复/复现；它不解释 4K 场景的低接受率，不能拿它声称加速收益。下一份真正新投机 profile 应显式按七槽检查，并验证相邻页边界、拒绝长度和接近 max_model_len 的路由。

后续先用短自然文本/随机文本和真实输入准备诊断区分工作负载效应与实现偏差，再接入已验证的 Q3 Split-KV。当前407-source非投机 profile 的 SpecNone 与42因果层假设不能直接关闭：必须原生计入42目标+5草稿 KV、scratch、共享头与新增 packed bank，保持标准 rejection 与真实概率。模型实测完成前不运行正式 N256 benchmark，不把组件提速换算成模型成绩。

所有代码、报告和数据只存容器/GitHub。旧实验冻结源未修改。此目录仅 CPU/源码审阅，不会自动启动 GPU。
