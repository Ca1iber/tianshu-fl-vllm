# 天数 BI-V150 32g：长请求prefill阈值1024没有吞吐收益

同期4K输入/1K输出、C64/N64，各3轮并预排首轮。只将实际CLI long-prefill-token-threshold从0改1024，全局预算2048保持；控制3525.328296→候选3520.442946 total tok/s，变化-0.138579%。暂不采用、不启动新Math或原正式benchmark。

|轮次|控制tok/s|候选tok/s|控制TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1314.276248|1308.878728|19103.442675|19278.537329|否|
|2|3522.377683|3525.946348|13296.513614|13453.006118|是|
|3|3528.278909|3514.939544|13317.644469|13469.961866|是|

384/384请求成功，逐轮seed配对及usage4096/1024/5120、原始文本已本地核验。7个greedy探针相同，不替代官方随机采样质量。TopP16/Decode KV32-w4-stage1/nativePrefill L0/B75G1及六份安装源保持相同，两组都实际完成75点预热；六轮没有新增调优记录。五份冻结源及实际server argv已核验，仅阈值不同；不混入IRRMS或RoPE候选。

当前V1 scheduler对running与waiting请求都使用long_prefill_token_threshold限制单次请求的num_new_tokens，再受全局预算约束。本实验测试请求间分配而非降低全局预算。此前全局预算1024下降8.27%，不能混用两项结论。本次阈值1024没有显示稳态收益，不断言所有调度参数都无收益。

全原始包4101712bytes/16分片，SHA256 181493ec56b95fb6e4cb00db8f478c21c8db3f06d938972e9c817a49145ba8f5。整包/分片、384请求、五份冻结源、实际TopP/Decode生产与合并记录、两组75点预热及均值均已本地核验。原件service-results/20261014_tianshu/prefill_fairness_service/20261014_prefill_fairness_service_tianshu。

正式最好3506.84 total tok/s、TTFT4494.12ms、独立Math103/105及tag tianshu-best-topp-w16-20261014保持。没有新16K，没有安装源/模型权重/原benchmark修改，沐曦只读。性能剖析：TODO有效图内kernel覆盖；接下来补测Decode KV32的1/2 warp启动配置，先数值/Graph后按ROI决定服务评测。
