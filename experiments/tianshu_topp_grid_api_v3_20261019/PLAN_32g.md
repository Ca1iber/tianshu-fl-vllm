# Top-P grid API V3：相对现用实现的回归门槛

候选GPU kernel、32..64批次路由、16Ktile/32grid上限、64行共享scratch，与V2逐字相同；只修正独立检查器的接受条件。V2真实失败结果和报告完整保留。

用冻结的原始assess函数 `require_strict=False`：保留logit值不变，且（与现用实现逐位一致，或通过原严格CPU排序/1e-6概率质量门槛）。严格CPU条件原样计算、记录和离线复核；报告分别统计严格通过、现用实现一致、基线自身非严格和仅因基线一致而接受的案例。不能据此宣称所有输入严格CPU一致。

13分布/批次×6概率×2图重放相位=156mask；21真实采样器全局/请求RNG状态；6跨batch共享scratch图生命周期；8原生fallback；260原始event。原Sampler及rejection alias不替换。完整归档下载、源与标量独立核验后才能写GPU_GATE，再运行原serve命令的4K/1K C64 N64对照各3轮，首轮排除。该短筛选不等于正式N256成绩，不自动采用或复用旧Math质量结果。
