# 天数量化探索，2026-10-22

当前已交付推荐版为不可变 **r3**：正式4K **5950.75 total /1190.15 output tok/s，TTFT3174.58ms**，较r2吞吐+0.6648%、TTFT-3.9694%，热轮极差0.2059%。新鲜完整105题C8=101、C64=102；903模型输入/73控制输入/原13serve参数和63环境全部通过独立最终审核。原checkpoint和结构不变，未训练/微调。

[r3直接Bash启动与复现](../tianshu_native_int8_attention_prefill_release_20261022/README_32g.md) · [r3正式报告](../tianshu_native_int8_attention_official_results_20261022/REPORT_32g.md) · [双并发完整质量](../tianshu_native_int8_attention_quality_results_20261022/REPORT_32g.md) · [三组同请求短测](../tianshu_attention_three_arm_results_20261022/REPORT_32g.md)。标签`tianshu-original-command-4k-native-int8-attention-prefill-20261022-r3`，提交9044c25，已push。只新增预填充QKV/O量化，M64保持BF16；额外378.74MiB全部计入真实显存。前序r2（5911.45）和r1（5777.22）均冻结保留，没有新16K成绩。

20:24后续：全42层M64组件已通过168工件/43008CPU校验，QKV1.204倍、O负向；[组件报告](../tianshu_native_int8_attention64_20261022/REPORT_32g.md)。仅增加QKV64的新版本915模型源/51控制源已冻结，正在完整C64质量评测，之后仅通过>95%才继续C8，不能借用r3质量。没有该候选的新吞吐或采用结论。

原生weight-only BF16接口12组/24相位确实支持，409600输出独立CPU界核验通过；[契约报告](../tianshu_native_weight_only_contract_CPU_results_20261022/REPORT_32g.md)。真实Down及整段MLP nativeW8A16性能测试已预备，等待GPU空闲，不将小尺寸API支持当作模型收益。只哈希native大库，不复制或上传。.so的float符号猜测已被BF16硬件证据纠正，条件FP32wrapper未执行。

每天北京时间06:00–24:00，本日硬停止23:58。代码、报告、关键结果在天数GitHub与容器，完整日志/数组/cache仅容器，无本地项目副本。GPU空闲窗口备份并清理1.15GB可重建FX缓存，所有903源/73控制和384strict缓存清理后再核验：[维护报告](../tianshu_storage_after_r3_20261022/REPORT_32g.md)。以下保留历史分阶段记录。

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

## 20:28：QKV64 新C64独立核验通过

新915源/51控制C64完整105题102正确，独立全量重计与QKV64capture42/O64BF16及真实prefill4096全42验证通过：[C64部分报告](../tianshu_native_int8_attention_decode64_C64_results_20261022/REPORT_32g.md)。该partial gate不能资格通过；新C8运行中，benchmark入口仍要求双批次独立gate，r3不变。

## 20:56：QKV64全模式质量失败，原生W8A16实际尺寸性能失败

新QKV64双105独立重计C64=102、C8=99（94.29%），不满足>95%，原冻结版本不重复抽样，不启动短/正式。C8真实M64非capture42bank/504Python调用，后续另立uniformdecode-only/小prefill BF16策略，不能声称已解决。[完整负结果](../tianshu_native_int8_attention_decode64_quality_results_20261022/REPORT_32g.md)。

小尺寸支持的nativeW8A16到了真实Down/M64，完整524288输出数值界通过但Graph9.928ms，对照BF16仅0.05940ms，整个MLP同样严重退化，不进入全42或Math。[真实原生W8A16负结果](../tianshu_native_Down_W8A16_20261022/REPORT_32g.md)。推荐r3不变。

## 2026-10-22 21:34：新上下文白名单质量与LM转换融合准备

当前已发布正式最好版本保持r3：5950.75 total tok/s、1190.15 output tok/s、TTFT3174.58ms，C8=101/105、C64=102/105，标签和发布证据不变。

915源全M64 QKV量化版本最终C8=99/105、C64=102/105，独立核验质量不合格，不重试、不跑吞吐、不采用。原生W8A16真实Down64虽然全部数值核验通过，但BF16约0.0594ms对原生INT8 NT约9.928ms（约167倍延迟），保留失败结论，不扩展到模型。

新928源版本限定QKV64只在FULL/64token/64req/uniform/noLoRA解码上下文，其他M64及QKV128均BF16。新C64全105题独立计分100/105=95.238095%，全部题计入且通过源/快照/上下文与Down核验；C8正在新跑。单C64部分门槛model_quality_qualified=false，不授权吞吐。已冻结的序列仅在新完整C8/C64均>95%后执行903对928短4K三轮排首轮；只有预设收益/稳定性/TTFT门槛通过才跑原N256四轮正式4K，再独立CPU核验，不自动发布。

另备LM64量化输出与Sampler FP32转换融合组件：原路径Triton GEMM BF16+torch.to(FP32)，候选显式保留BF16舍入再直接写FP32；另比较原生INT8+dequant。真实r3 binding显示LM64在eager，故按eager>3%收益筛选，graph只作诊断。当前仅CPU准备，尚未运行GPU，必须等当前序列结束且GPU空闲。不声称模型收益。详情见tianshu_lm_quant_cast_20261022的冻结计划。

用户确认每日06:00至次日00:00可用，今天GPU硬停止2026-10-22T15:58Z（北京时间23:58）。代码、报告和关键结果存天数仓库/容器，完整日志、数组与checkpoint保留容器，不存本地，不推送大原始数据。

## 2026-10-22 22:59：QKV上下文方案收益不足，LM补齐进入双质量验证

928源QKV64上下文白名单新C8=103/105、C64=100/105均通过，但短4K热轮平均5883.57→5887.40 total tok/s，仅+0.0651%，未过预设收益门槛，不跑正式、不采用。报告保留在tianshu_native_int8_attention_decode_scope_short4k_20261022。

LM64输出融合组件将实际eager延迟约0.7187降至0.5751ms；进一步将M32..63补至64行后，真实BF16回退对照约1.0036..1.2339ms，候选约0.5742..0.5751ms。仅组件线索，不是整机吞吐。详见tianshu_LM_pad64_20261022/REPORT_32g.md。

新925源模型只改LM M32..64，保留r3所有QKV/O策略、原模型权重、13启动参数及63环境项；M<=8仍BF16。C64全部105题答对103，但旧LM计数审计错误导致原实验audit_failed。原记录不修改，独立CPU恢复重现原失败、更正实际计数对象并重跑全部运行时/计分检查，恢复103/105。详情和来源见tianshu_LM_pad64_audit_recovery_20261022/REPORT_32g.md。新C8仍在运行，完整双质量通过才执行903对925短4K，显著稳定正收益才条件执行正式4K；不得把单批次或组件结果称为新最佳成绩。

当前正式已发布推荐仍为r3：5950.75 total tok/s、1190.15 output tok/s、TTFT3174.58ms，C8=101/105、C64=102/105。
