# 天数BI-V150 32g：CUDA Graph输出非持有别名验证

已读取实际FL _C_ops_registry：weak_ref_tensor fallback为lambda t:t。原图管理器意图以弱引用释放图输出，fallback保留强引用。83点Prefill PIECEWISE在mem0.85和0.70分别1/83、2/83后OOM。此为待因果验证的框架内存假设，不直接声称根因已确认。

独立C++仅以ATen from_blob生成共享data_ptr/shape/stride/dtype/device、析构不释放原存储的别名，不计算或修改张量。它仅适用于原始存储由CUDA Graph私有池保有的输出；不是任意张量的所有权替代。先CPU三种stride验证原owner可回收且不解引用已释放CPU存储；随后同池四种CUDA Graph shape、变输入各十次重放、最后按捕获顺序重放逐位正确。未接入模型，未修改安装源。

探针尚未编译运行，任何服务启用必须先通过以上生命周期与重放检查，再新Math和正式验证。
