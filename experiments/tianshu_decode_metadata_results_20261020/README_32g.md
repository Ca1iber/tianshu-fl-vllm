# 元数据融合4K服务评测进行中

17:06启动，原件`/workspace/benchmark-results/20261020_decode_metadata_service4k_tianshu`，同279份源码/54环境、唯一融合开关。run_4k_screen.sh直接串行control/candidate，每边3轮64请求，预排除首轮；23:58北京时间硬截止。

verify_screen.py复核实际请求、token、时序、配对输入、来源/环境、原Graph、未来KV准入以及新方法绑定和64批次编译记录。需要候选真实B64命中，不把开关或微测倍率当作服务收益。最终Math和正式4K/16K必须使用最终选定组合的相同源码/环境；旧正式Math不借给本候选。

后续优先级：完成本次4K后选最终组合；若融合有稳定收益则保留，反之保留单独未来KV准入。随后新Math Level3/105严格>95%，再进行原始双负载正式确认。本元数据候选不重复跑16K小规模筛选。
