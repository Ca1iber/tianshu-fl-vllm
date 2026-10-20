# 未来输出KV预留：服务实验进行中

先跑4K/1K C64 N64，两个方案各3轮、首轮排除。run_4k_screen.sh只在上一轮已完成且GPU空闲时启动，所有服务有2026-10-20 23:58北京时间硬截止。原件根目录`/workspace/benchmark-results/20261020_future_kv_service4k_tianshu`。

verify_screen.py复核相同260来源/52环境、唯一方案开关、原始命令、真实Graph、实际KV容量/原生调度以及所有请求。跨配置固定seed文本作为诊断，组内重复必须稳定；质量未经新Math验证之前始终quality_pending=true、adopted=false。正在运行不等于已经通过，不把该背景组合取代正式发布版本。
