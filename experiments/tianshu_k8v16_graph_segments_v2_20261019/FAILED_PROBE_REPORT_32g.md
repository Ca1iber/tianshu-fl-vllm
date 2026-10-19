# BI-V150 32G：Graph诊断 V2 失败记录

结论：audit_failed，未采用，没有新性能/Math声明。Breakable 类在worker中被导入，但本配置未启用，所以 __call__ 仍为0。ModelRunnerFL还导入了独立 vllm_fl.compilation.graph.GraphWrapper，这是本配置实际使用的包装器；此前错误地将导入类视作实际路径，V3直接接入FL实现。

原始根目录 /workspace/benchmark-results/20261019_k8v16_graph_segments_v2_tianshu。全部145负载请求与3固定种子请求成功；204冻结源/43环境/K8V16页和GPU退出审计通过。实际35×42=1470个attention捕获调用；无Breakable图对象，关联审计失败，不意味没有图。

档案6188208bytes，SHA256 9fc2170f90cdf70ffa1b90ea05b86cffdef982e79b9d83e92d7ba290c365ac07，24分片与完整SHA均已本地验证。local_failure_audit.json复核全部源、请求、KV布局和校验文本。保留原始失败状态及运行代码。

相较V1仍无正式性能结论。已验证最优4997.05 total tok/s、公开baseline2028.01以及Math100/105均不变。下一步V3挂FL实际GraphWrapper，未采集GPU profile。
