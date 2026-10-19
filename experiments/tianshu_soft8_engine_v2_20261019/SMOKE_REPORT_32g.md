# Engine V2真实服务短试跑通过

来源`4960ce317758ec581c19ac68bf2dc8a65fa583ee`。官方serve argv保持，当前最佳38环境项加软件8位mode/记录目录共40项。35个FULL decode图已完成捕获并启动API；实际42层FullAttentionSpec与KV view均为UInt8，page8192，容量1020368 tokens。共享arena536887296 bytes和页表131072 bytes已在profile_run、determine_available_memory内分配计入。大批次2D T16/w1为245regs、0spill；首次T128/w16图捕获OOM在该版未重现。

两次C1固定seed canary文本一致。8个512输入/128输出请求全部成功，4份metrics零抢占；这是一轮短服务检查，不是4K benchmark、性能对照或Math质量。一个_index_linearized_jit_function首次调优记录，约0.08秒；没有把它说成零调优。157冻结来源与生成codec、全部8响应/usage/输入seed/时间、cache/路由/环境在本地复核。

目前同一源运行C64/N64、4K/1K的3+3服务对照，第一轮排除。有明显提升再做新的105题Math与256请求正式4K。正式最佳4093.81/Math102不变，量化版尚未采用。

原始包SHA256 `03462c9e3880e4e9a381684fafe3a29b731700ed645c08c1db94558189eda30f`，21分块与整包验证。
