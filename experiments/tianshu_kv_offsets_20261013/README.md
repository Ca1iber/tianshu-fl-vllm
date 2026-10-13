# 天数 KV 写入候选：整数位宽与启动参数分别验证

8场景×6配置的48组GPU scatter/完整缓存逐位一致与CUDA Graph检查已通过；整数32本身多数场景无明显收益，Prefill启动参数有微测收益。尚未在实际模型服务替换或采纳。见 `results/20261013/micro_32g/REPORT_32g.md`。

实际 `TritonAttentionImpl.do_kv_cache_update` 调用 vLLM 的 `triton_reshape_and_cache_flash`。
原实现每 token 写两个 128 维 KV 头，7 个 stride 参数和载入后的 slot index 使用 int64；CUDA 默认 num_warps=16、num_stages=10。

`kv32.py` 从冻结的真实源码生成 process-only int32 元素偏移版本。公开 slot_mapping 张量仍为 int64，原浮点复制、负数 padding 的忽略逻辑、64 位物理指针不变。
入口检查所有带 tile 余量的元素偏移和缓存总 slot 数均小于 2^31。调用者仍须满足原生 KV 写入 API 的有效 slot 合约：-1 padding 或缓存内合法槽位；GPU probe 会独立检查每个 slot 值。

`probe_kv.py` 对原64/新32分别测试默认 w16/s10、w4/s1、w1/s1。8 个场景覆盖 Decode 与 2048-token Prefill、负数 padding、QKV stride2560、交错 K/V cache，以及真实 32682 页缓存容量。逐位对比完整 scatter 结果和未改动缓存，捕获/重放 CUDA Graph，轮换计时顺序。这能区分位宽收益与启动参数收益；不能将微测直接当服务成绩。
