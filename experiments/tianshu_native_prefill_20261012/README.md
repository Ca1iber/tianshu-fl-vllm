# 原生 GQA 预填充 tile 筛选

原生统一attention按KV head组打包Query head；默认D128的BLOCK_M16、BLOCK_Q2及小KV tile未针对天数预填充负载筛选。候选保留原生算术，扩大BLOCK_M并相应调整BLOCK_Q和二维网格，不能只改变BLOCK_M而保持旧网格。

独立进程测试六种M64/M128/M256、KV64/KV128、w8/w16、s1/s2参数组合，直接以正式最佳FA2 BM128/BN128/w8/s2为对照，保持随机分页、真实交错KV缓存及非连续Query。全量FP32、选定CPU oracle和输出尾部检查，交替CUDA Graph计时。五种场景包括2048/2048、2048/4096、4096/4096、ragged多请求及2048/16384；这不是完整16K benchmark。

尚未跑，不声称已发现收益。GPU串行排在M2048 GEMM服务对照后、小GEMM复核前。数值、实际服务和对应Math105未完成前不采用。
