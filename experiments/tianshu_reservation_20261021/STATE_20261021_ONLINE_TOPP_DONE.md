# 2026-10-21：真实 GEO 基线扫描合并已完成，无新增模型成绩

承接TOPP_BASELINE_CORRECTED。当前5586.28正式4K、TTFT3780.63、Math102/105推荐不变，无当前新16K；8192formal5608.40未采用。

新方向tianshu_topp_online_sum_20261021确实以实际已采用GEO09156e66为对照，仅合并两次max/sum扫描。120CPU数学/源反转检查、12GPU场景×2模式×2阶段通过，独立CPU完整1444行/188528640float严格1e-6语义通过。主要B64组件耗时变化−.3065%到+.3321%无明显收益，不跑此候选服务/Math/完整benchmark。GPU runner209722 terminal，exit0，真实GPU已释放。结果报告/关键CSV/JSON在tianshu_topp_online_sum_results_20261021。

重要记录器失误：原阶段input.pt在flip后被覆盖，原CPU离线checker失败原样保留；冻结GPU程序不改/不重跑。独立CPU恢复目录tianshu_topp_online_sum_cpu_recovery_20261021核验冻结源码一次flip+文件相等并精确逆flip恢复原阶段，完整输出重新计算全过；GATE明确恢复标志。不能说原记录器成功，不能拿此CPU恢复当新GPU试验。

已回答用户最近方向为框架预算、Spec输入/接受率与采样组件，明确重复warp没有新增收益以及当前无新模型成绩。后续继续新候选，先读实际报告非陈旧README避免重复。当前QKV/O/Down仍BF16；DownNN组件大M仅小收益、decode负收益。当前407源INT8只GateUp大M/NN decode与LM64，可能探查其他projection的真实权重INT8可行性，但先检查历史与质量风险，不直接改推荐版。

Goal仍active。预约到10-22 00:00，北京23:58/15:58Z GPU硬停；SSHmaster04:27建立16:27前刷新。容器/GitHub唯一存放，用户2个dirty/Ops4cdc3b7不动，无子agent，下一轮优先N64小4K；每完成benchmark轮报告。观察timeout不重启，任何GPU-started py/sh/CPU_GATE冻结。
