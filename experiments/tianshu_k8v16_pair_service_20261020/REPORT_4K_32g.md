# K8成对读取：4K N64服务+2.5671%，进入新质量/完整4K验证

2026-10-20，BI-V150 32G。在已正式通过Math103/105的future-KV组合上，唯一模式差异TIANSHU_PAIR_LOAD_MODE；候选仅在3D decode中选择已通过逐位组件检查的pair_rows K读取。原始serve argv、4096预算、83预热点、128MiB暂存区、max_num_seqs64、async、K8/V BF16、S16、writer/reducer、严格原生TopP与RNG保持。两臂同282份冻结来源/54项实际环境，Graph捕获和真实B64模型重放与新kernel SHA关联通过。

|模式|第1轮：预排除|第2轮|第3轮|后两轮均值 total tok/s|稳态极差/均值|
|---|---:|---:|---:|---:|---:|
|现用K字节读取|4740.0649|5043.3447|5036.6332|5039.98898|0.13316%|
|两字节pair_rows|4762.3141|5172.5585|5166.1802|5169.36935|0.12339%|

相对同规模控制+2.56708%。每轮4K/1K C64 N64，独立请求/usage/计时/metrics/源码/配置审计全部通过，384/384成功、零抢占。候选三轮无新增autotune。这个5169.37是小样本筛选，不是N256官方正式成绩；也不能把组件6.75%当模型提升。两臂seed777 canary逐位文本一致只作诊断，不作额外准确率门槛。

初始CPU替身stride接口与首次Worker初始化的dispatch-owner身份错误各自留有失败源码/日志。原始服务目录20261020_k8v16_pair_service4k_tianshu没有成绩，不重写；修正身份定位后的独立v2目录完成本表，并生成independent_screen_gate.json。原始完整回复和运行记录留/workspace/benchmark-results/20261020_k8v16_pair_service4k_v2_tianshu；GitHub只存本目录代码、DIAGNOSTIC_4K_GATE_32g.json和DIAGNOSTIC_4K_ROUNDS_32g.csv。

正式5103.41/3885.91、Math103/105暂保持。后续先完成word先解码/warp2组件，选定实际有收益的最终配置后跑新Math（严格>95%），再跑完整4K N256；本筛选没有新Math或16K正式结果。
