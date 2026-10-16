# 天数 BI-V150 32g：异步调度开关对照准备

当前默认async_scheduling=True，尚未在天数做独立开关对照。假设：Iluvatar驱动与框架的同步/元数据成本可能改变异步调度的实际收益；不假定关闭必然更快。当前GPU预算16384实验仍独占运行，本方向仅CPU准备、不排队启动。

已读容器实际vllm源码：config/scheduler.py中async_scheduling为bool或None，False禁用；engine/arg_utils.py对bool使用BooleanOptionalAction，生成--no-async-scheduling。候选仅追加这一标志，仍为4K输入/1K输出/C64/N64、budget4096/83warm、CV/KV16/S8/原生TopP、max_num_seqs256，保留prefix cache。每组新启动3轮、预排除首轮，验证真实WorkerFL记录的异步布尔值，保存384响应并离线复算。

当前没有GPU服务结果或新的Math，不采用。只有显著收益且复测稳定后才准备新Math与正式4K成绩；旧最佳4091.29/Math103of105不能当成本候选质量证明。
