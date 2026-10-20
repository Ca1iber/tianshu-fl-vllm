# 成对 K 读取：完整官方4K新正式成绩

2026-10-20，原始 vllm serve 参数与原始 benchmark 脚本运行4096输入/1024输出、C64、每轮256请求，共4轮，排除首轮。正式均值为 **5235.52 total tok/s、1047.10 output tok/s、平均TTFT4161.08ms**。相对上一版5103.41提升2.5887%，相对公开基线2028.01提升158.1605%；TTFT比上一版4197.36降低约0.8644%。

|轮次|total tok/s|output tok/s|mean TTFT ms|成功请求|计分|
|---|---:|---:|---:|---:|---|
|1|5094.71|1018.94|5794.36|256|排除|
|2|5233.62|1046.72|4165.89|256|计分|
|3|5239.43|1047.89|4157.19|256|计分|
|4|5233.52|1046.70|4160.17|256|计分|

计分三轮极差/均值0.1129%。1024/1024请求成功、零失败、零抢占；未来KV虚拟预留最终全部释放。独立 verifier 对原始CSV、benchmark标准输出、排首轮算术均值和当前状态交叉复算，没有改写原始comparison的 adopted=false。独立采用证明另存OFFICIAL_GATE_32g.json。

同一新配置重新跑官方 MATH-500 Level3 全部105题，103题正确（98.095238%），105个请求自然结束、无错误无截断，严格通过大于95%的要求。完整回答与逐题原始数据留在容器；关键评分和校验在 ../tianshu_pair_quality_20261020。

变化为把相邻两个K编码字节以uint16一起读取，在进入原有矩阵点乘前分别重建BF16值。保留K编码、V BF16、分段、warp1、FP32累加及采样路径。此前组件96项完整GPU检查与256项独立CPU检查均通过，成对读取结果与当前K8/V16位级一致。标准N64小样本对照5039.99→5169.37（+2.5671%）；本报告采用N256正式成绩，不拿小样本替代。

保留Future-KV准入、小暂存区128MiB、max_num_seqs64、4096预算、83预热点及现有长上下文分段策略；不包含metadata融合、word/uint32读取或W8A8整数GEMM新实验。审计绑定282份运行来源、54项实际进程环境及8份控制文件，验证官方C64的真实Graph捕获和模型回放都使用f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002；质量评测另验证C8路径。

本配置只完成正式4K，没有跑完整16K。此前16K3885.91属于Future-KV前版，不能当成本配置的16K成绩，也不据此宣称新配置已完整满足两个负载的资格要求。

启动入口为 ../tianshu_pair_release_20261020/serve_tianshu_minicpm.sh，直接Bash/jq导出配置并exec原始13项argv。CPU入口演练通过282份运行来源与8份控制文件SHA、实际54项环境、污染开关清理及带空格记录路径；这不是额外一次GPUbenchmark。26份生成Python源码已随Git保存，模型与编译缓存未加入Git。

原始性能目录：/workspace/benchmark-results/20261020_pair_K_official4k_tianshu。
原始质量目录：/workspace/benchmark-results/20261020_pair_K_math_tianshu。
前版正式目录：/workspace/benchmark-results/20261020_future_kv_official4k16k_tianshu。
