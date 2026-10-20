# RoPE与K8/V16写入融合：CPU准备，尚无GPU或模型收益

针对当前MiniCPM的16个Q头、2个KV头、128维Neox RoPE，尝试把RoPE和随后K8/V BF16缓存写入合成一次kernel启动。每个token/头一组工作，输出正常BF16旋转Q/K，同时将旋转后的K编码和原V写到相同12KiB物理页；负slot仅跳过KV写入，Q/K仍输出。

当前服务custom_ops=['none']，CustomOp.dispatch_forward选继承的forward_native，不能拿FL forward_oot里的另一种FMA/舍入实现作基线。组件保留原生四次BF16乘法的中间舍入，再做减/加和BF16存储；沿用现有软件8编码，不改变RNG或缓存格式。

CPU_GATE_32g.json为14场景CPU布局/舍入检查：B1/8/32/40/41/63/64，两种QKV行stride2560/2688，包含位置131071、负slot、分散页及页末token、非零物理storage_offset。与真实RotaryEmbedding.forward_static逐位一致，完整缓存及前后guard字节、输入不变检查通过。它没有运行Triton kernel，不能当作GPU数值、Graph、服务或正确率验证。

旧单独RoPE融合服务仅约+0.31%，本假设只有合并KV写入后才可能增加价值。下一步GPU组件必须覆盖服务实际编译后的原生RoPE加现用writer，避免用eager多kernel基线夸大收益；变数据重放同Graph，核对全部Q/K及物理字节、输入/RNG不变，再看计时。当前GPU由未来KV正式4K/16K占用，组件尚未运行，也未安装任何模型hook。只有组件有实际收益才设计4K N64服务测试。
