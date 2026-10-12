# BI-V150 32G：短提案目标验证及完整draft骨干数值检查

实际native metadata builder/forward与交错paged KV：5个target causal场景（Q2/B1、Q3/B2跨4096/4097、Q4/B8、Q3/B64、混合Q1/2/3），3个保留Q7的draft非因果场景（KV7/B1、KV4096/B8、KV4097/B64）。随机physical page、非连续Query/KV、完整FP32 oracle、输出sentinel尾部及CUDA Graph检查均通过。81项选定CPU向量交叉检查全部通过。

执行hook仍采用L0；target短验证使用M128/KV64/w8/s2，draft非因果保留原tile。验证脚本通过可控causal元数据测试双向draft，而不是用causal oracle替代真实draft语义。所有原始case/error/sample字段保留。

fixture分别对同一个选定launcher调用两次，记录的两组时延不能用于宣称优化收益。此处只有数值证据；2步服务性能/质量另行验证，正式最好不变。没有新增benchmark分数。

完整原始包1869265 bytes，SHA256 9cfdd6a34d42e067664f9f1ab9ca2c6e0ada88c0022b10f0652c411ffbd4be30已分块及整包本地核验。执行后补存本轮验证脚本与原始fixture；执行hook从同期7步服务冻结源码复取，记录采集说明，避免把后续2步代码当成本轮执行版本。

性能剖析：TODO，未采集profiler。原始权重、安装源码、官方benchmark和沐曦仓库保持不改。与先前Q8的35项验证相同，这些是oracle检查，不代表端到端提速。
