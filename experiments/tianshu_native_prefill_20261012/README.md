# 原生 GQA 预填充 tile 筛选

原生统一attention按KV head组打包Query head；默认D128的BLOCK_M16、BLOCK_Q2及小KV tile未针对天数预填充负载筛选。候选保留原生算术，扩大BLOCK_M并相应调整BLOCK_Q和二维网格，不能只改变BLOCK_M而保持旧网格。

独立进程测试六种M64/M128/M256、KV64/KV128、w8/w16、s1/s2参数组合，直接以正式最佳FA2 BM128/BN128/w8/s2为对照，保持随机分页、真实交错KV缓存及非连续Query。全量FP32、选定CPU oracle和输出尾部检查，交替CUDA Graph计时。五种场景包括2048/2048、2048/4096、4096/4096、ragged多请求及2048/16384；这不是完整16K benchmark。

初版30项有12项数值失败：零KV prefix扩大tile后出现首query半幅输出。保留初版失败数据；使用独立生成源码对无sink在线softmax分母显式初始化0后，30/30场景及24/24 CPU检查通过。具体低层编译指令原因尚未证明，sink初始化及安装源码未修改。

当前已验证配置为M256/KV128/w16/s1、显式无sink L0、原有adaptive Decode和预算2048。C64/N128、4K/1K短对照吞吐+8.199003%、TTFT−26.513501%；独立Math102/105（97.142857%）、零请求错误；官方4K/C64/N256四轮排首轮均值2712.55 total tok/s、542.51 output tok/s、TTFT4720.38ms，四轮1024请求全部成功。相对公开baseline吞吐+33.754271%、TTFT−59.214000%；后3轮吞吐极差/均值7.777174%，未把最快单轮当作最终成绩。

完整质量及正式数据已逐块、整包和原始计数本地核验。质量标签tianshu-verified-native-prefill-20261012，最佳标签tianshu-best-native-prefill-20261012；启动入口serve_tianshu_minicpm.sh。完整4K结果和证据见results/20261012/official；未运行优化版完整16K。GPU继续串行做后续筛选，尚未验证的GEMM/预算/融合候选不包含在该最佳启动配置中。
