# 软件8位KV prefill tile独立筛选

前置真实pipeline V2已完整下载核验：16场景、编码/写入/metadata/forward正确，B64 decode含writer约快5.7%，但原BF16 prefill256/128/w16参数在Soft8中慢32倍。本轮只改prefill BLOCK_M/TILE/warps，codec、writer、BF16 Q/out、FP32累加、decode路由不变。

15配置覆盖M32/64/128/256、tile32/64/128、warp4/8/16，stage1；CPU直接执行实际Producer类，全部网格/BLOCK_Q/launch参数通过。只先做prefill2048与带4K历史的chunk2048，含变Q/K/V、适用的长度/slots同图重放、完整物理writer/storage、padding/scratch/RNG与量化后FP32/CPU参考。计时包括实际编码写入和forward，各7组×20；每个配置记录实际n_regs/n_spills、grid/tile/warp/stage，未测前不把溢出当根因。

前置GPU结果SHA和全部冻结来源绑定在PREREQUISITE_32g.json，旧pipeline/runtime不改。本轮仍是组件，不是服务或Math；有足够收益并完成全边界复核后才考虑真实engine spec hook、完整缓存管理和新质量。
