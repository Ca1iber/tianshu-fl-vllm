# 首次IXKN采集失败，原件保留

2026-10-22，Down/M64 W8A16候选。应用在采集前后核对完整BF16输出，均与已独立审核的相同artifact逐位一致，应用complete；外层ixkn-cli退出139，串行队列停止，未继续后面7项。

导出的metrics.ixkn-rep为60KB。两次独立导入（CUDA_VISIBLE_DEVICES空/正常）均退出255并报告空device name，暂不可用，未引用计数器。应用正常不表示采集正常，原失败状态/日志/导出不改。

新V2从启动监听、跳过10次预热取第11次、直接CSV，退出0，CPU审核105行/8section，编译binary与原组件完全相同。V3保留context元数据、预热时stop、目标start/stop区间，原生BF16也成功；两种新方法使用新目录。未确认原崩溃的唯一根因。

原raw `/workspace/benchmark-results/20261022_ixkn_down_M64_W8_channel_candidate_tianshu`；有效V2/V3数据见对应目录，原队列仍保留EXIT139。
