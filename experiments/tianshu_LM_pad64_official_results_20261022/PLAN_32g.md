# LM补齐正式4K独立CPU核验

925模型输入/实际完整控制器及快照、原r3 903源、原checkpoint SHA、原13 argv/63环境。独立重新运行真实正式全运行时审计（新LM64 guard>=100、LM8 BF16、GU/attention/QKV-O/Down/KV/scheduler/sampling/RNG、无在线tune/零抢占），原四轮N256 CSV/汇总排首轮，1024成功请求。新C8原82控制器、新C64 CPU恢复质量及原失败来源SHA、额外恢复控制器全部核对；不能把原C64失败说成complete，模型数值未改，分数不重测挑选。

晋升条件正式总吞吐>5950.75*1.005、热轮极差<1%、TTFT<=3174.58*1.01及全部质量/运行时/结束GPU空闲。CPU保持CUDA未初始化。大比较JSON留容器并无损gzip，Git仅关键Gate/CSV/报告，不声称16K或全赛事资格；另行发布通过前不采用。
