# 整数PV GPU原语验证通过，尚未测完整attention性能

实际Corex/Triton GPU原语5组M/T：16×32、16×64、16×128、32×32、32×64，输出D128，2warps/1stage。每组3个改变输入、V符号和scale的Graph重放相位，分别验证diagnostics=False/True，合计15相位、两种诊断模式。

独立CPU重新从保存的原始P/V/Vscale推导IEEE FP32 RNE和Python整型点积，逐项核验 **19968个实际Pq、43008个实际INT32输出、86016个两种模式实际FP32输出字**。量化scale也逐位相同。所有输出/未使用诊断buffer/尾部guard、原始输入和CPU/GPU RNG保持。每个编译版本的cache hash、二进制SHA、寄存器/spill/shared记录在PRIMITIVE_CASES，完整Tensor仅留容器。

CPU源生成另验证48个context/tile/segment地址组合，候选大tile页表load采用tile_mask，覆盖672个原未mask可能超出最小页表宽度的尾部索引。默认False的原V8访问路径保持，修改可逆恢复精确父源。

这些结果证明实际整数PV原语的RNE/整数累加/FP32恢复符合独立参考，不是完整attention输出、KV生命周期、模型质量或速度的证明。没有计时、接engine或跑PV的新Math；不采用。

下一步在独立目录测试真实writer+K8V16/原V8/整数PV完整attention、partial/strided物理页和改变输入Graph，对CPU FP64参考检查，再测实际资源与耗时。只有组件有收益才考虑engine/cache/prefill集成与fresh Math>95%。当前GPU优先跑预算8192的原N256正式4K，不并发组件。

原始路径`/workspace/benchmark-results/20261021_k8vi8_pv_int8_primitive_GPU_tianshu`，runner `.../20261021_k8vi8_pv_int8_primitive_runner_tianshu`已complete并确认GPU释放。原语在旧质量队列因元数据后处理失败而退出后执行，完成后才恢复正式4K，GPU任务无重叠。
