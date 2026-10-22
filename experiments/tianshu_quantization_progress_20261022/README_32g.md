# 天数量化探索，2026-10-22

当前已交付推荐版是不可变 **r2**：正式4K **5911.45 total /1182.29 output tok/s，TTFT3305.80ms**，较r1吞吐+2.3234%，热轮极差0.2008%。新鲜完整105题C8=102、C64=103；全部890模型输入与原13serve参数/63环境通过独立CPU核验。原checkpoint保持不变，没有架构转换或微调。

[直接Bash发布入口及复现](../tianshu_native_int8_down_prefill_release_v2_20261022/README_32g.md) · [r2正式4K报告](../tianshu_native_int8_down_prefill_official_results_v2_20261022/REPORT_32g.md) · [r2双并发完整质量](../tianshu_native_int8_down_prefill_quality_results_v2_20261022/REPORT_32g.md)。发布标签 `tianshu-original-command-4k-native-int8-down-prefill-20261022-r2`，提交5bb7f7d。没有新16K成绩，未声称整个比赛资格已审核。

19:42更新：新QKV/O预填充候选903项模型输入、额外397135872字节W8副本，已通过独立完整105题C8=101、C64=102；[质量报告](../tianshu_native_int8_attention_quality_results_20261022/REPORT_32g.md)。同请求短4K对照净增0.8720%，热轮均值5880.27vs5829.43；[三组归因报告](../tianshu_attention_three_arm_results_20261022/REPORT_32g.md)。相同框架BF16控制5804.93，量化相对它+1.2978%，不能将所有收益归于一个kernel。该候选正进行原始N256x4/exclude1正式4K，尚未替换r2。

后续独立M64全部42层组件与原生权重量化接口仅在前一GPU服务停止后执行；需要新数值、性能和完整质量证据才可采用。每天北京时间06:00–24:00，本日硬停止23:58。代码、报告和关键结果在天数GitHub与容器，原始数组/完整日志/cache只在容器，无本地项目副本。

第一阶段不可变 [FINAL_REPORT_32g.md](FINAL_REPORT_32g.md) 属于较早r1（5777.22），不是当前r2。[第一阶段负结果](../tianshu_quantization_survey_20261022/REPORT_FINAL_32g.md)局限当时W4/W8原型；后续阳性证据分别见对应版本，不覆盖历史结论。以下为按时间追加的研究记录。

推荐版作为冻结检查点保留。后续继续探索其他M范围的原生INT8收益，全部新尝试使用独立目录，未通过新模型与质量门槛前不改推荐版。本索引允许更新，独立版本的最终报告和数值源码仍冻结。

## 14:40 后续量化研究（r1推荐版不变）

- [32点M曲线](../tianshu_native_int8_m_curve_20261022/REPORT_32g.md)：真实三类基线，32点/96阶段独立复核。多数中大M组件正向；4K主路径4096已经采用，因此未声称新的整机提升。
- [LM64原生INT8](../tianshu_native_int8_lm64_20261022/REPORT_32g.md)：全部8配置数值逐位一致，最快仍慢约0.3%，反量化抵消GEMM收益，不采用。
- [Down单层](../tianshu_native_int8_down_20261022/REPORT_32g.md)及[全42-bank MLP](../tianshu_native_int8_down42_20261022/REPORT_32g.md)：候选完整MLP在M64/512/4096分别+8.55%/+13.28%/+20.47%，504份阶段CPU独立复核通过；这是新Down近似，尚无新的模型质量或服务吞吐。
- [新Down服务设计](../tianshu_native_int8_down_service_20261022/PLAN_32g.md)和[新质量验证](../tianshu_native_int8_down_quality_20261022/PLAN_32g.md)：890模型源、27控制文件冻结，原877源完整保留；原13argv/63环境。新增约504MiB单一NN packed Down bank计入真实显存profile。14:38启动新scopeall C64完整105题，之后才跑新官方C8；若质量不足，另测预先声明的prefill-only范围。当前运行结果只在容器，未宣称通过。

正式r1仍为5777.22 total tok/s、TTFT3459.44ms，fresh C8/C64各101/105；原不可变FINAL_REPORT_32g.md与r1标签不变。

## 15:30 更新

新Down allscope实际C64为95/105（90.48%），拒绝，不跑C8/吞吐。原audit_failed保留：[负结果](../tianshu_native_int8_down_all_negative_20261022/REPORT_32g.md)。新V2恢复实际BF16 vendorMM回退并保留原三项GEMM检查，仅prefill量化；首次启动遇到50GiB配额，尚未评测。经快照内容去重和可重建FXcache备份，恢复约7.4GiB空间，17.8万snapshot文件SHA完全一致、r1缓存/启动检查通过：[存储维护](../tianshu_storage_recovery_20261022/REPORT_32g.md)。

正在使用同一冻结V2代码、新raw根重试prefill C64：[重试计划](../tianshu_native_int8_down_quality_v2_retry_20261022/PLAN_32g.md)，原始配额失败目录未改。只有新C64/C8均>95%才运行与精确r1基线的短4K对照。正式r1仍5777.22 total tok/s。

CPU通道平衡仅使相关验证误差相对减少2.43%，没有独立随机holdout/模型质量结论，暂不占GPU：[CPU分析](../tianshu_down_channel_balance_CPU_20261022/REPORT_32g.md)。

## 15:42 之后：Down prefill-only V2 和通道分拆

保持已发布 r1 不变。Down 全范围 INT8 的旧候选仅 95/105（90.48%）已拒绝；只在 M128/512/4096 启用的 V2 首轮 C64 已完整通过 103/105（98.10%），原三个 BF16 GEMM 预热路径也通过启动检查，正在继续 C8。尚无该候选端到端吞吐成绩。磁盘配额故障的原始失败目录保留，当前是同一冻结数值/控制代码在新输出目录重试。

新增 `tianshu_down_outlier_partition_CPU_20261022/REPORT_32g.md`：BF16 128 通道加 INT8 主体的 CPU 相关验证误差 2.5630%，原 INT8 为 3.4638%；这是合成样本数值线索，尚未验证 GPU 性能或模型质量。

## 16:10：激活 K 分块缩放的 CPU 线索

新增 `tianshu_down_group_activation_CPU_20261022/REPORT_32g.md`。不改原 W8，激活 K128 分块的相关验证误差为 1.7794%，K512 为 2.2106%，原逐行 K6144 为 3.4638%。全部 504 份旧工件的原始量化对照独立逐项重算通过。此处没有 GPU 速度或 Math 结论；分组缩放需要新融合 GEMM，不能直接套一次整体 native int_mm。

## 16:11：Down prefill-only V2 完整质量通过

独立原始评审重算：C8 102/105（97.14%），C64 103/105（98.10%）。C8 本轮未实际命中 INT8 Down，C64 已核验 M128/4096 全42层真实调用，M64 仍 BF16。两份完整105均计分，C8 两个输出上限请求未剔除。详见 `tianshu_native_int8_down_prefill_quality_results_v2_20261022/REPORT_32g.md`。4K 短对照已启动，尚无候选吞吐结果，已发布 r1 不变。

## 16:47：Down prefill V2 正式4K独立核验通过

正式四轮 total tok/s =5731.84（剔除）/5912.19/5917.01/5905.14；汇总 **5911.45 total、1182.29 output tok/s，TTFT3305.80ms**。相对 r1 吞吐+2.3234%、TTFT-4.4412%，热轮波动0.2008%。1024请求全部成功，零抢占、服务就绪后无autotune，890模型输入及全部控制快照和继承运行核验通过CPU重核。新鲜 C8=102/105，C64=103/105。详见 `tianshu_native_int8_down_prefill_official_results_v2_20261022/REPORT_32g.md`。正在独立 r2 打包和CPU dry-run，原r1不覆盖，无16K复测。

## 17点后续组件探索

已封存 r2 `tianshu-original-command-4k-native-int8-down-prefill-20261022-r2`，提交5bb7f7d；正式5911.45 total tok/s、TTFT3305.80ms，所有CPU核验和/root、/tmp污染环境dry-run通过。六份必要控制JSON及完整comparison均精确gzip保存，合计压缩871448 bytes，无预测全文或数组上传。

`tianshu_down_outlier128_gpu_20261022` 首版侧路128在42-bank独立随机GPU输入上数值通过但速度负优化：整段MLP M64=0.142750ms，BF16对照0.134484ms；M4096=4.298262ms，普通native INT8为2.128494ms。不进入Math。此结论限当前固定tile实现；全输出误差3.5309%→3.0924%，未声称模型精度。

`tianshu_down_W8A16_CPU_20261022` 保留激活BF16、仅W8的理想CPU相关验证误差1.2534%，原W8A8为3.4638%，需要实际GPU数值与速度验证。动态K分组GPU初筛已启动，原r2不覆盖。

## 18时后组件探索：实际服务对照及新投影

- `tianshu_down_BF16_control_20261022` 核对实际 M64 Down 为 vendor mm，Graph0.05959ms，torch.mm0.07435ms。旧单层1%收益不能当作对当前服务收益；r2正式端到端不受影响。
- `tianshu_down_group_gpu_v4_20261022`、`v5` 均数值通过，但改用实际BF16对照并包括GateUp后无收益，不接模型。
- `tianshu_down_W8A16_gpu_20261022`、`v2` 全输出独立FP64/预先数学误差界通过；高warp最好Down0.07846ms，对照0.05899ms，负结果保留。
- `tianshu_native_int8_attention_projections_20261022` QKV/O预填充原生INT8有组件线索，M4096相对显式vendor BF16约+39%/+31%；仅组件、非模型成绩，尚需实际路由核验。新 `tianshu_native_int8_attention42_20261022` 正验证全42层，M128/512/4096，M64暂不接入。所有新尝试不覆盖已发布r2。

## 18:36 新 QKV/O 预填充量化进入全量质量验证

`tianshu_native_int8_attention42_20261022` 和实际 eager apply 对照 `tianshu_native_int8_attention42_v2_20261022` 均通过504阶段/84真实bank的GPU与CPU独立核验；M4096组合组件分别约+35.96%/+36.71%。eagerF.linear进入vendorlinear，而已有生成代码将linear降为extern_kernels.mm；已区分两种对照，未将组件数值当作模型吞吐。

`tianshu_native_int8_attention_service_20261022` 冻结903模型输入，原890完整保留；新增397135872bytes单NN布局权重计入真实DeviceMemoryProfiler。只接QKV128/512/4096与O512/4096，M64等BF16vendorMM回退，原三项2048实际GEMM检查保留。`tianshu_native_int8_attention_quality_20261022` 冻结51控制输入与原13argv/63环境，C64新鲜105题先跑，>95%才继续C8；无新模型成绩或采用结论。
