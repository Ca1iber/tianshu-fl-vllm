# IR-native RMSNorm与A8量化融合组件

当前真实服务日志显示custom_ops=['none']、rms_norm/fused_add_rms_norm优先级native；实际模型RMSNorm.forward_native调用vllm.ir.ops层规范。因此baseline不能用RMSNormFL.forward_oot/FlagGems插件路径。本实验用安装版本IR原生impl_fn独立torch.compile(fullgraph)后接当前动态A8 quantize作为组件对照；明确不声称与完整模型生成CUBIN一致。真实模型checkpoint/结构不修改，无训练或微调。

固定新kernel：FP32输入/残差加法，FP32方差与rsqrt，按IR规范先舍入BF16再乘BF16权重，最终BF16归一化值再精确A8逐行scale=max(abs(y),1e-8)/127、div_rn/nearbyint/saturate[-127,127]。功能式返回norm/residual/q/scale，输入/权重不修改，无全局Tensor workspace。K2048、eps1e-6、4warps/1stage固定；不现场搜索/修改数值算法。测试M8/64/128，residual有/无，真实layers0/20/41的input与post_attention两个norm权重，三个独立固定种子/尺度/零与抵消行，共108数值case。

每个case完整保存输入/权重、对照和候选输出；候选A8/scales对其实际BF16 norm的独立CPU重新量化必须逐位一致。完整K的FP64 norm参考验证预先数学界：8*gamma(2K)+32*eps32相对误差，加两次BF16舍入的ULP界；零参考要求精确零。记录对照/候选norm与q/sc差异，不要求独立编译器实现与候选逐位相同，也不据此虚构原模型数值不变。由于可能改变编译器舍入行为，未来模型仍需新完整C8/C64各105题>95%。越界guard、输入/权重/RNG状态须通过。

所有Graph返回Tensor对象单独保留，写入毒值后replay必须恢复原完整输出；不能只检查eager Tensor。计时对照为独立编译IR-native norm后接原A8，候选为固定融合kernel，eager和Graph各五轮固定warm20/replay200，首轮排除，取余下四轮中位数。主筛选只看M64两种residual Graph延迟均下降>3%；eager及M8/128诊断，不作为模型吞吐。计时前全部compile/warm完成，无其他GPU工作；108case全数值/独立CPU核验通过才是组件线索。

GPU必须等当前keep128完整序列7900/9812结束且无服务、GPU空闲，不能与质量/短测/正式并行。所有源/计划/数值规则首次GPU前冻结，失败保留且不原样重跑取幸运值。今日2026-10-23T15:58Z硬停止；大数组/编译cache仅容器，GitHub仅代码/报告/关键结果，本地不留副本。组件不自动接模型/发布。
