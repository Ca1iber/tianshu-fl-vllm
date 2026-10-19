# BI-V150 32G：Graph诊断 V1 失败记录

结论：保留 audit_failed，不给性能或新Math结论。所有145个4K/128诊断请求和3个固定种子校验请求成功，但所挂 upstream CUDAGraphWrapper.__call__ 无实际调用；因此不能建立图对象到重放的关联。该现象不表示没用CUDA Graph。

原始根目录 /workspace/benchmark-results/20261019_k8v16_graph_segments_tianshu。实际worker trace_85073.json 包含1470个attention捕获调用（35×42）与133组launch，但 graphs=0/calls=0。43项环境、204冻结源以及真实 K8V16 布局审计通过。独立本地复核全部145请求token用量/种子、3校验文本、所有源SHA、KV审计和GPU退出，original_phase仍audit_failed。

完整档案6178173 bytes，SHA256 f09a3e0be450a5be8bd236134935a7292913ae63cb82f19e0878587b91b4f5d0。本地 service-results/20261019_tianshu/k8v16_graph_segments_v1_audit_failed。

与上版/官方baseline：该仪表化短请求无可比正式分数；已验证正式最佳4997.05 total tok/s（官方2028.01）与Math100/105不变。下一版改挂实际框架类，不修改本轮运行源、CPU gate或失败状态。未采集GPU profile。
