# 组合候选质量/正式评测准备

新增独立run_validation.py及validation_profile.py；从3轮筛选实际/proc中捕获的39项白名单配置复现服务，只重映射输出目录，同时清除继承的其他实验开关。拒绝5+5未完成、收益不足2%、稳态重叠、冻结源改变或组合未通过新Math的正式评测。

启动前后校验23服务源、9安装源及本轮验证脚本/原官方benchmark，完整快照；实际131预热、budget16384/maxseq64/async、CV/KV16/S8、严格TopP重新渲染、融合采样、native Prefill、int32 gather及零preemption均查原始记录。Math按官方105条Level3/C8/温度1/p.95/max32768；正式原脚本4K/1K/C64/N256，四轮排首轮，不改原脚本。Math C8与服务/正式C64的覆盖范围分别注明。

CPU已经用实际已完成screen原始记录走通新的runtime审计（不导入Torch/无GPU）；用历史105条完整Math预测和review走通中性结果解析，含index35达到max_tokens但正确的案例。配置复现及排除其他开关已检查。以上CPU检查均不是新的组合Math或正式成绩。当前仍在独立5+5确认，尚未部署/执行新验证脚本。
