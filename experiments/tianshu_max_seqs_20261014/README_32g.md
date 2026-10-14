# 天数BI-V150 32g：max_num_seqs64框架对照

待运行的新鲜3+3服务实验：以完整新Math105及原官方4K通过的.cv配置为双方共同基线，保持budget4096/实际83点、S8、原生TopP与NativePrefill、1/.95、4K/1K/C64/N64，每边三轮，排除首轮。候选只增加--max-num-seqs64，不删请求、不改变输出长度或模型上下文上限。CPU实际嵌套argv与Popen拦截验证通过；WorkerFL构造后记录真实max_num_seqs、budget、异步调度、模型长度及当时的Graph捕获列表，不把源码默认256当作实际运行证据。

当前CPU检查使用显式worker fixtures，真实Loader AST验证control256/control128/candidate64和错误candidate128拒绝，保存的fixture值不是天数服务观测。没有新GPU吞吐分数，没有把预分配减少宣称为提速。

run_after_quality.py等待现有.cv质量链的实际进程结束、阶段complete、完整官方归档存在且CV正式优于旧最佳，GPU空闲后才串行启动。若上游失败/质量不通过/正式不改善或租期不足则不启动，不并行挤占评测。新控制器冻结17份源，当前质量链23源不改。参数作用可包括worker/modelrunner预分配与dummy/Graph准备范围；即使冷启动节省时间，也需独立计分轮次证明稳态吞吐。

已准备独立5+5确认入口run_confirm_pair.py（未启动），必须显式传--screen-root，实际3+3完整384请求结果须>=2%且排除首轮后两组不重叠才允许进入；两边全新服务，预先排除前2轮，保留后3轮，共640请求。run_confirm_trial.py及本目录foundation_confirm_trial.py冻结实际5轮参数，verify_confirm_cli_cpu.py已验证真实嵌套argv、Popen拦截、两组预算4096与仅候选max-num-seqs64；CPU检查不代表运行时吞吐。verify_downloaded_confirm.py核验640原始请求、实际worker配置及.cv源。当前3+3尚未启动，不将准备工作写成收益，原17冻结等待源码不改。
