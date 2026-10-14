# 天数BI-V150 32g：0.70显存下83点piecewise仍OOM

双方同0.70显存/4096预算/83预热点/.cv/S8/KV16/原生TopP，control完成3804.459674（排除）/4061.779243/4054.130611，后两轮4057.954927total tok/s。实际KV401280tokens。candidate在piecewise捕获2/83点后OOM：申请42MiB时仅24MiBfree，零性能轮次；不是完整吞吐对照。原始192个control请求用量/TTFT/吞吐与17冻结源、实际OOM及GPU清理本地复核。

原始1187118bytes/5parts，SHA256 8d57616c2bcda6ebfebc3466d3ad9ac8fd05af1ae40bc382e70c824cba8842dd，raw在service-results/20261014_tianshu/piecewise_memory_oom/20261014_piecewise_graph_memory_service_tianshu。无新质量/正式成绩，当前best4091.29保持。

后续源码发现FL weak_ref_tensor fallback为lambda t:t，图输出强引用可能导致跨shape存储无法释放。独立非持有别名CPU/Graph验证通过，但仍需模型因果验证；不能把该假设当作全部OOM根因。模型profile未采集。
