# 13:57 T16 warp/stage扫描完整拒绝，继续检查K/V加载顺序及配对K整数地址

预约10-22 00:00止，GPU23:58硬停。正式推荐不变：521c154/tag tianshu-original-command-4k-prefill-int8-20261021，407源/62env/原13argv，4K5586.28total/1117.25out/TTFT3780.63ms，Math102/105。TopP99失败不复测，Down SplitK、grid-only、integerPV拒绝；QKV prefill432源六轮384请求+.293182%未达事前>.5%，不做Math或正式，6c4775e报告已推送，extra exact407参考GPU未启动。

新tianshu_pair_t16_pipeline_20261021已complete，GPU/CPU均结束，runner236233终态Z。7路current w1s1/w2s1/w4s1/w1s2/w2s2/w1s3/w2s3；只n>24变，均T16与原S/.cv。196完整GPU参考、14077952元素离线复核、304独立CPU向量/2128项原CPU参考、882时序组、全部输出逐位相同。4K当前.552635ms，新六路1.323351/2.016669/1.634649/1.994093/2.237259/2.387253，全负收益。stage2/3真实透传，旧Producer硬编码stage1，已在最终JIT前覆盖且4928 CPU stub+实GPU metadata确认。regs176→132/121/141/122/128/116但全0spill、速度更慢；不能按寄存器数推收益，不做服务Math。报告/GATE/CSV待本次推送。

下一方向尚未GPU准备：可能四路因子对照 current / addr32_only / late_V_only / both。当前f1df读核确实有tl.int64步长、physical_block_idx.to(int64)、reducer宽offset；旧address32实验针对7b94...S16旧K8V16读核，kv32/all32在4K−.086%/−1.026%，未采用。source.py生成AST只改int64→32，guard为signed元素offset（含2048×sumstride保守masked余量）<2^31，不改pointer和slot_mapping。当前f1df配对K后尚需find/Git确认是否已有组合，不可直接重复旧源测试或把旧失败抹去。

late_V是假设：当前f1df循环在QK和softmax前计算v_offset、load/cast V，考虑把V offset/load/cast推迟到P/alpha之后、PV之前以缩短活跃区间；更晚加载也可能损失prefetch，不能凭寄存器估收益。只对3D/puredecode/KV_NONE/Kuint8/VBF16/TD关闭的明确范围启用，保留其它分支/reader ABI/数学操作。用可逆精确文本替换+AST验证改动边界、默认/未选分支和完整numeric/bitwise检查。组合all32必须基于新f1df而非旧7b94，AST纯dtype变更另验证。

注意旧component地址guard只选择producer，new组合若窄化reducer，应让producer/reducer同一guard结果回退wide，避免输出stride大时producer回退但reducer仍窄。guard应测试合法/宽offset/不支持mode和实际返回/grid/args，wide仅CPU边界不分配>2Gi-element GPU张量，不冒称GPU宽缓存覆盖。已有native.Producer在address32/component_launch.py，source_proof规范化AST保持；别改父文件。

元数据复用已只读核对：天数builder已共享Qcu；真实paged FlagGems ops/attention.py:1284接口只用seqused_k/block_table，cuK仅非paged，Python分支分配empty_like(cuQ) dummy但不K cumsum/D2H，沐曦5217a6d解决的每层K累计D2H不在本路径。可忽略微小dummy分配，不当作大收益。FP16 GEMM已60fe829/tianshu_gemm_dot_dtype_20261019完整负收益，非Attention FP16实测，不能再称FP16从未做过。

GPU当前空闲。所有东西容器+GitHub，不留本地，用户dirty gateup_silu两文件不碰，FlagGems clean4cdc3b7，沐曦只读，无子agent。SSH master/tmp/tianshu-20261021-control，16:27前刷新。raw仅/workspace/benchmark-results。
