# 量化组件 IXplorer 计数器准备

已核对容器自带 `ixkn-cli --help/--list-sets`，工具提供详细 instruction/memory/occupancy/scheduler 等组和 cudaProfilerStart/Stop 区间。此前只读取帮助，未采集 GPU 计数器。

profile_one.py 只重放已通过独立 CPU 审核的 NT候选真实角色/shape最优tile；从该GPU原始artifact读取同一phase2 activation，重新只读载入原weight、打包并检查完整候选输出逐位一致。BF16/W8A8控制读取同artifact的已保存selected输出，W8A8整数结果另外准确比较。只有这些检查通过，才在 profiler start/stop 内执行目标kernel一次。候选计时不包含diagnostic store；BF16/W8A8保持其实际路径。Profiler结果只用于成本归因，不能替代Graph吞吐或Math。

运行方式（仅前任务和审核结束、GPU实际空闲后）：ixkn-cli --profile-from-start off --set detailed --cache-control none --csv -o <新raw路径>/metrics --log-file <新raw路径>/profiler.log python3 profile_one.py ...。不覆盖旧报告，工具失败/权限错误也保留。导出大报告只在容器，GitHub保留脚本、摘要和关键metrics。每任务15分钟上限，23:58前停止，原weight与数值源不改。

本目录尚未执行 profiler，没有计数器结果/根因结论；NT数值/性能审核未完成前不得启动。
