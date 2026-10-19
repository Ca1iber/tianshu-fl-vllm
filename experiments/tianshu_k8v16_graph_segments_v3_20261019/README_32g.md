# 天数 32G：实际 FL GraphWrapper 捕获与重放诊断 V3

Local diagnostic, not a performance score.

V1 挂 upstream CUDAGraphWrapper、V2 挂已导入但本配置未启用的 Breakable，均完整请求完成而 graph审计失败。保持两者失败原件。V3 直接挂 ModelRunnerFL 导入的 vllm_fl.compilation.graph.GraphWrapper；源SHA ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2。该类独立于前两类，内部持有 concrete_graph_entries 与 entry.graph，以实际 torch.cuda.CUDAGraph 单图 capture/replay。

保持已验证 K8V16 数值/引擎配置，43项环境中只额外仪表化；复用C1/8/16/24/32/64各4K/128短请求，总145请求+3已验证固定种子。记录每个 descriptor 对应实际GPU图对象ID、捕获时native42层配置、重放时CPU现有computed长度。无GPU Tensor读取与新RNG操作。只通过全部图-捕获-重放关联、KV/源/env审计才给诊断结论。

CPU检查执行下载的实际 FL __call__ AST，覆盖capture、replay、NONE、无context、mode mismatch、原始参数/返回对象/异常转发。无性能或新Math声明；正式最佳4997.05/Math100/105不更新。未采集硬件profile。
