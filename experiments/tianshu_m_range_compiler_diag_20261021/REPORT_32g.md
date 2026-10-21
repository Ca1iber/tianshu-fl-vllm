# 首次分段融合服务启动失败：诊断占位张量改变SME编译特化

v1控制已完成三轮，热轮5546.14 tok/s；候选启动在第一项M128 binary exact检查失败，未测速，Math/C64/正式均未开始。原始两个服务目录及失败JSON、日志全部保留。

CPU读取真实Triton缓存元数据：组件SME mask83，实际服务mask19；除hash外唯一编译选项差异为use_sme。读取并实际执行安装版get_corex_sme的纯形状分类函数（meta tensors；明确模拟torch.corex=True的实际GPU平台身份）：它按每个张量形状/dtype决定SME位，所有24M均证明差异来自未启用诊断时仍参与JIT特化的Projected参数。组件使用完整二维BF16矩阵，服务使用tiny dummy.view(BF16)，丢失第6位。

修复新service_v2：仅在diagnostics=False调用时，复用已分配二维BF16输出作为未使用Projected参数，无新增显存、无全局SME开关；对应mask恢复83。原Binary SHA严格匹配组件检查保留，实际服务启动再次验证每个M，不以CPU推导宣称GPU成功。不能强行传use_sme选项：安装版JIT的miss路径会从bound_args重新计算覆盖此参数。原v1不改源重跑。
