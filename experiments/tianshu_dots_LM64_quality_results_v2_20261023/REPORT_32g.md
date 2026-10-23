# Down64 + LM64 W4：新鲜完整双质量通过

同934模型来源/95实际GPU控制器：C8原版本正常complete，102/105=97.142857%；C64原105题101/105=96.190476%，严格>95%。全部210题reviews/predictions/report和完整分母、raw SHA、934来源/95控制器及快照、checkpoint、实际13argv/63环境重新独立CPU核对，完整profile与所有accuracy计算字段逐字重算一致。

C64原controller因继承Down审计遗漏的LM64 INT8期望而audit_failed。原失败文件/原冻结来源不改、未重跑C64 GPU。单独CPU校正仅将该错误期望改为真实W4A16，并要求LM64 W4记录>=10、旧W8 LM64记录为空，原所有其他Down/GU/KV/attention/scheduler/sampling/LM8/内存/capture/主模型图重放检查保留；原失败SHA、新CPU校正代码SHA和CPU_recovery_complete结果独立绑定。门禁显式记录C64_original_audit_failed_reconciled_CPU=true、original_experiment_passed=false、GPU_rerun=false，不把原失败改成原通过。

实际Down42个W4捕获、旧Down64 BF16捕获0，C64与真实主模型M64执行链接；新LM64为图外真实native W4调用，至少512次已记录，不冒充图重放。新增LM142049280B及Down280756224B保留副本、实际load profiler warmup/native profile scratch/所有模型存储计入通过。QKV/O64仍BF16、LM8仍BF16；C8的单个M64记录不当作LM64请求覆盖，LM64资格以C64完整新鲜模型评测为依据。原checkpoint/模型结构不变，未做训练或AWQ校准。

完整QUALITY_GATE_32g.json记录独立双资格、所有原始SHA与额外CPU校正来源；KEY_SCORES为两轮各105行关键结果。大答案/日志/数组只在容器。尚无本组合吞吐成绩或采用决定，正式推荐仍r3 5950.75 total tok/s。下一步CPU准备全部结束后运行新short4k_v2，避免计时轮次后台CPU干扰。
