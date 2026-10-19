# BI-V150 32G：原命令K8V16默认预算4096全新Math通过

最终框架默认值实现独立重新评测Math-500 Level3全部105题，102/105=97.142857%，严格高于95%。105次请求全部成功，零错误、零缺失；105份回答均自然stop，没有max_tokens结束。错误索引为18、25、87（原始记录的零基索引）。不复用显式CLI4096版的102/105或旧2048版的100/105。

使用官方generation配置：temperature=1、top_p=0.95、max_tokens=32768、batch=8、timeout=3600；全部回答正常保留和判分，没有选择性重试或缩短长回答。总生成阶段约20分钟，最后两条长回答仍持续生成，最终也自然结束。没有采集硬件profiler数据。

serve argv保持官方原命令，框架实际默认预算由2048改选4096，max_num_seqs=256、async=True，预热83点；候选选择仅作用于受支持天数MiniCPM5-2B单卡配置，显式用户预算优先。模型、Q和输出仍BF16，K用固定scale1/16的软件有限E4M3，V保留BF16，42层实际12KiB页与有界预填充暂存和直接字节decode均核验。默认值包装不读取或改变RNG，采样器仍原算法。

完整下载原始预测、判题、服务日志、205份冻结源和43项实际进程环境，在本地重新执行verify_validation_downloaded.py：105份prediction与review记录、运行时布局和最终统计均通过。原始档案6,517,835 byte、25分片，SHA256为c20e0f9544731d83a541f8da65b201ccb0ade4cd3989a7499e8b73776951801a。证据位于service-results/20261019_tianshu/k8v16_budget_default_math/20261019_k8v16_budget_default_math_tianshu，local_validation_audit.json已生成。

这轮是质量门槛验证，不是吞吐得分。N64筛选提升3.27%后，本质量通过才启动同205份源、同43项环境、同官方serve argv的原脚本N256正式4K四轮，首轮排除；该正式测试正在运行，旧正式最好4997.05 tok/s仍保留，不跑16K。
