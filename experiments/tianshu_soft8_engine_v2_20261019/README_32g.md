# 软件8位KV engine V2：大批次纯decode单独配置

V1真正UInt8 spec/allocator/profile成功，但FULL图捕获的B256/B248落到BM256/BQ32/T128/w16二维prefill配置（908 spills）并报Triton OOM。V2保留失败全部原件和134测量来源，重新建立独立runtime；prefill staging/codec/有界fallback和B<=64三维decode不变。

对max_seqlen_q<=1的纯decode，在超过原64序列3D门槛时使用二维BM16/BQ2/T16/w1，不扩充segment scratch。状态由真实call的max_seqlen_q设置，不以q.shape==num_sequences猜测，支持capture padding。CPU检查B80/128/248/256的实际Producer，新增GPU B80/128/256真实writer/metadata/forward，计划21场景，检查2D时整个segment scratch保持。资源与数值通过才重新启动服务。V2同时绑定实际安装的WorkerFL/ModelRunnerFL源码及仓库mirror，完整服务之前不运行正式benchmark。
