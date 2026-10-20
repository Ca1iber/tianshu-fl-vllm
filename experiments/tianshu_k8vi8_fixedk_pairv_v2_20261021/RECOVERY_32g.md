# KV V1编译失败的独立V2恢复

V1原件 /workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_GPU_tianshu 保留，phase=failed、0完成case，三个旧路径编译通过后，PairV因 `stride_v_cache_0/1/2` 是原生runtime整数而不能放进tl.static_assert失败。失败队列保持failed，不将编译通过冒充数值/性能通过。

V2相对失败source.py只删除这一行不可静态求值断言，其他地址/量化/softmax/PV算术不改。对应偶数正stride、D128、末stride1、mode2/int8条件在实际Producer调用Triton之前用真实Tensor/stride元数据检查，并记录host_pair_stride_guard_passed。CPU对三组有效stride及七组无效参数验证守卫，全部65536有符号word与8原生布局仍全测。新目录/新raw/新GPU进程，不改V1冻结来源。

四路和原nativeV8一致布局、真实writer、七轮20 Graph重放及三次改数据参考检查保持；GPU结果仍待完成。新CPU检查必须CUDA_VISIBLE_DEVICES=-1。只在GPU空闲时运行run_and_audit.sh，截止仍2026-10-21T15:58Z，无模型/正式分数预设。
