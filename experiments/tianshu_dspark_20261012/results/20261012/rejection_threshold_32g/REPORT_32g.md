# BI-V150 32G：随机拒绝采样 FP64 故障与正 FP32 兼容修复

当前正式最好仍2712.55 total tok/s、TTFT4720.38ms，独立Math102/105。该组合没有投机拒绝采样，因此不受此次故障影响。

此前投机服务随机采样长期显示100%接受。固定索引修复后的4K/1K、C64/N128曾记录5254.726695 total tok/s；4K/128短输出曾记录8133.276910。独立Math已开始，因拒绝率异常主动中止，保留部分预测，不形成完整准确率。后续正式benchmark未启动。

独立数值诊断直接调用安装版原始 `rejection_random_sample_kernel`，用已知概率、给定阈值、固定恢复/bonus token构造CPU逐项决策。覆盖单batch、不同草稿长度、含0长度请求三组。

相同输入采用FP32阈值时3/3逐位匹配CPU；采用FP64时3/3失败，GPU把本应拒绝的token继续接受并输出bonus。给定FP64值的GPU/CPU往返逐位相等，不能仅归因于主机数据传输。原始 `generate_uniform_probs` 产生的512个FP64样本还全部为0，unique=1。已证实的是当前CoreX软件栈的这两条路径失效，不推断具体底层编译器指令根因或所有FP64操作的行为。

因此所有此前随机投机服务性能数据失效，不能据其宣称有效吞吐提升。相关历史报告追加更正，原始数据不删除；attention、RoPE、row-gather等独立逐位/CPU检查结果仍可复用。此前对原始sampler源码未改动的陈述并不足以保证实际采样语义正常，此次新增直接数学判定验证。

修复仅替换进程内uniform阈值生成 helper：使用FP32 `torch.rand`，保留每请求generator和跳过0长度请求的规则，最终1-u映射至(0,1]，避免0阈值接收target概率为0的proposal。原始概率比、target logits处理、恢复采样及拒绝kernel保持原实现；不启用synthetic接受、不强行接受、不修改target权重。

这是正FP32离散uniform的数值兼容实现；随机数精度和具体随机流相对原设想的FP64有所变化，不宣称逐随机数等价。准确率仍需独立105题严格>95%，更改阈值不会免除质量门槛。

功能验证48/48逐位匹配CPU，覆盖B1/2/3/64、长度0/1/3/7、有无draft概率、混合greedy请求、三种种子及每请求generator，含target或draft概率为0。给定边界阈值的原FP32三组检查也通过。65,536样本最小3.4570694e-6、最大0.9999944、unique65279、均值0.5002273、低于0.5比例0.5014648，支持正常正随机阈值；不是完整统计分布证明。

新开关 `TIANSHU_DSPARK_FP32_THRESHOLDS=1`、runner `--fp32-thresholds`，记录实际阈值dtype/范围、无synthetic和无概率比修改。随机投机性能、Math和正式4K均强制该兼容开关。质量/正式审计要求相同开关和冻结rejection_compat.py源码。proposal记录明确标注阈值helper兼容改动，不再笼统说target sampler完全不变。

真实服务重跑已进入4K/1K、C64/N128首轮，实际64个FP32阈值非零且不恒定；19:03日志平均接受长度约1.22、平均draft接受率约3%。这是首轮中间统计，不是完整吞吐结果。重新评估性能，不能沿用失效的5254.73作为收益门槛。

原始故障与修复诊断、被中止质量数据共三包均已分块及整包SHA256核验备份，清单见同目录backup_manifest_*.json。原始rejection sampler SHA256为953bb182ac2e0ef8a2c07b9ac3e157e3542f3159b3ae6f3f3f5378acfe851a86。当前修复源码和可复现probe随本提交保存；诊断归档的Git bundle早于未提交兼容代码，完整测试代码以本提交的probe/rejection_compat为准。

性能剖析：TODO，尚未采集修复版profiler。GPU任务串行，使用租期至23:58；不修改原始benchmark或沐曦仓库，不运行优化版完整16K。
