# 天数原始页表上传与slot mapping成本

使用实际安装BlockTable/CpuGpuBuffer、固定page16、最大请求256、实际原方法与pinned CPU，比较活跃1/8/64请求的8192列和320列控制实例。8192列来自131072上限/page16的源链推导；320列只用于复制成本对照，未改完整模型上下文配置。先验证整行及padding相等，真实slot mapping含尾部-1，改变CPU页号与GPU positions、清空输出后重放slot Graph两次。

FlagGems启用record/once，与WorkerFL rank0相同；交错7组，每组20次，分别测整行commit、slot mapping与二者的host wall及GPU stream event。微测不直接作为服务收益。只有证据显示值得优化，才设计保留所有add/append/clear/move/swap/直接写入/异步与padding行为的增量上传。

当前仅准备源码。无模型服务、Math或正式benchmark；无新16K，23:58硬停止。
