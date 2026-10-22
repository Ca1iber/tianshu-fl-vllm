# 保留context元数据的目标区间采集

V2候选直接CSV采集退出0，105行/8个section及应用前后输出/同一binary校验通过。V1导出损坏不引用。

V3使用新的profile_one.py副本，CLI从启动监听，在torch.cuda.init()之后先profiler.stop()，完成所有warmup/数值核对后才start，目标操作后stop。这样原生BF16的libtuner预热次数可变也不会被误作正式目标。数值计算/输入和独立审核artifact保持；前后校验仍必须通过。从启动监听的设备元数据与目标区间一起保留，直接CSV输出。

先仅Down/M64原生BF16，过滤mm_kernel；卡实际空闲再启动，最多15分钟，23:58截止。原件不修改，失败保留。必须应用完成、工具退出0、完整CSV及实际kernel属性核对后才能引用。未采集前不宣称原生与候选根因比较。
