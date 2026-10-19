# BI-V150 32G：FL GraphWrapper 运行路径范围限定诊断

全部145个4K/128负载请求、3固定种子请求成功；35个实际FL GPU图对象，1470个attention捕获调用；42层capture配置、204源/43env、实际K8V16页、零抢占、GPU退出已独立下载复核。

FL GraphWrapper 是实际调用类，独立于前两版挂的 upstream/Breakable 类。每个descriptor持有一个实际graph对象；本轮未走Breakable分段路径。所有捕获max_seqlen_k=131072，B1/2/4/8/16/24/32/64分别使用S64/32/16/16/8/16/4/4等长上下文配置，与已验证policy一致。没有发现误用短上下文S4的问题，不修改捕获参数。

**范围限制**：原始controller仍audit_failed。C64负载输出仅128token，prefill期间先到的请求已经开始完成，最终实际观测活跃decode数为1..63，没有exact64活跃请求；严格审计要求{1,8,16,24,32,64}所以失败。已见descriptor64的同一GPU图重放63个活跃请求。没有补造exact64或将原始状态改为complete。新增audit_observed_scope.py只验证实际已观察范围并写独立local_observed_scope_audit.json，不替代旧严格审计。无需为此重复整套GPU诊断。

实际根目录 /workspace/benchmark-results/20261019_k8v16_graph_segments_v3_tianshu。档案6202597bytes/24parts，SHA256 fbb8b90e3359439b68e9f646eb5cadeccd32ff86d14d6b4c6c88834975e3f53b，完整及所有分片已校验，本地 service-results/20261019_tianshu/k8v16_graph_segments_v3_scope。

这是运行路径诊断，不是正式分数或新Math。当前正式best4997.05 total tok/s、公开2028.01和Math100/105均不变；未采硬件profile。下一步转向缓存读取提示与scheduler实际预算对照。
