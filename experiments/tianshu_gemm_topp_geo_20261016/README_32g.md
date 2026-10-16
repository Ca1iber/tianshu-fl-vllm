# 天数GEMM与TopP32K几何搜索组合

原始官方serve argv的4K N64 3+3轮对照：3940.09→4028.84 total tok/s（+2.252538%），候选后两轮差距0.156492%。384请求与56份源码已下载复核，见[短测报告](results/20261016/screen3_32g/REPORT_32g.md)。

全105题Math已完成102/105（97.14%），105成功/零错误。17:58原103条候选预测全部原样保留，新增两条并重新评分全105题；68源/38环境/原参数/数据指纹/原103字节不变均已本机复核，见[完整质量报告](results/20261016/resume_actual_32g/REPORT_32g.md)。

正式4K N256×4已完成，排除第一轮后4093.81 total /818.76 output tok/s，TTFT4309.35ms；相对上一版3992.97提升2.525438%，后三轮极差0.155112%。1024请求、58源、38实际环境、75暖点和原始argv/benchmark SHA均下载核验，见[正式报告](results/20261016/official4k_32g/REPORT_32g.md)。本版本作为已验证的4K新最佳；未运行该版16K。

在容器以 `bash experiments/tianshu_gemm_topp_geo_20261016/serve_tianshu_minicpm.sh` 复现同一38环境和原始serve argv；实际启动前检查58份源SHA。脚本的shell检查及实际dry-run对照通过，未以该脚本额外启动另一服务。CPU调用诊断在正式服务退出后独立运行，不作为吞吐成绩。
