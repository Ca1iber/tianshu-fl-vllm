# 下一项 RMSNorm→A8 融合：真实调用边界核对

当前 NN 服务控制组的实际 EngineCore 配置 custom_ops=['none']、ir_enable_torch_wrap=True、IR rms_norm/fused_add_rms_norm priority=['native']。因此不能只看到插件里的 RMSNormFL.forward_oot 或 FlagGems fused_add_rms_norm 就假设它们是正在编译的归一化路径；模型接入前仍须实际 FX/capture 调用证据。

实际 native IR 计算源码 /usr/local/lib/python3.12/site-packages/vllm/ir/ops/layernorm.py：x/residual 先转 FP32 相加，residual 另行舍入回 BF16，variance 使用未舍入的 FP32 和；rms 乘积先转 weight.dtype，再乘 weight，最终转 orig_dtype。融合必须保留这些 BF16 舍入点，不能直接拿 FlagGems 另一套 FP32 连乘公式替换。现有 A8 还要求 div_rn、nearbyint/RNE、clamp[-127,127] 与每行 scale；不能把 norm 单体精度/量化单体精度合并当成模型质量证明。

当前真 INT8 linear 是 opaque torch custom_op，其 Python 实现内部才 quantize→GEMM，FX 看不到量化节点。已装 vLLM rms_quant_fusion 主要匹配 IR norm→现有 FP8/C-extension quant；单开 fuse_norm_quant 不会自动融合本项目 opaque INT8 路由。可行方向是先把量化节点显式暴露给 FX，再做天数支持的融合或模型块显式接入；必须同时处理动态 M、残差别名和不支持 M 的 BF16 fallback，避免只测 component 后假定图中命中。

本次只做源码与实际配置核对，没有启动 RMS GPU 组件，没有改动框架或当前运行服务，未作性能/数值/模型收益承诺。先完成 GateUp NN 服务和大 M prefill 组件，后者若有收益优先新的模型质量与正式 4K。

源码 SHA256：
- IR layernorm d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910
- model layernorm 696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c
- custom_op 4b4ba01623005d6a93698eaec007a934dd512bf9ec7241090927de819621de2e
- rms_quant_fusion d4fc85f6e2ef74cec2987c590968c18e24b98c4ea54e2c4d8ac19d4015834d19
