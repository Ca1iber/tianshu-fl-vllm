# 天数 BI-V150 32g：首轮尾部冷停顿的CPU栈诊断

原服务N64/C64/4K1K首轮258秒、后续101秒；int32计数与缩图也出现类似首轮停顿，GPU利用率低而Engine CPU忙。Triton autotune打印为零，首轮16份排序/扫描编译元数据集中最初6秒，不能解释后半停顿。

诊断仅添加Python daemon thread每秒用sys._current_frames在GIL下读取file/function/line，不读取locals，及时释放frame引用；模型、原算法和采样参数未改。两轮复现首轮257.031269秒、第二100.983722秒。diagnostic_only=True、performance_score_allowed=False，这些吞吐不作为优化成绩。

依据API每10秒日志划分第一轮中的high（generation>500且running>=60）/low（generation<100且running>=16）时间窗，选择最近下一个日志窗口、最大12秒间隔，将主线程栈对应到同一窗口。高速60样本中54在ModelRunnerOutput.get_output等待GPU输出；低速119样本中116有SQLAlchemy栈、96直接经过build_sql_model_by_db。可核查原始epoch/完整栈见stack_summary_32g.json与全量归档。

低速链：model_runner._prepare_input_ids→FlagGems index→按实际batch形状生成index_linearized wrapper→LibTuner缓存get→SQLPersistantModel.get_sql_model→build_sql_model_by_db→AutoBase.prepare(engine)→反射整个SQLite数据库/构建ORM映射。源码models/sql.py:148没有限制反射表，而每个新index wrapper都有不同模型表名。因此首轮动态批次变小时，CPU反复做全库schema reflection；已有配置命中不会打印autotune，日志零调优不能排除此成本。

结论：数据库反射是本轮低速尾段的明确CPU热点。待后续以相同请求测试只反射当前表/缓存映射方案，验证是否真正消除约156秒额外成本；尚未实现、没有修复A/B，不把完整时间差全归为已量化净收益。按用户最新要求，诊断后已优先进入其它int64→int32排序缓冲实验。

尝试记录：外部py-spy0.4.2仅装remote/tmp，容器无SYS_PTRACE无法读栈；v1 C faulthandler异步转储在服务导入时段错误-11，GPU已清理，完整失败源码/日志本地cold_stacks_failed_v1保存。不能把导入期失败归因模型。随后v2 Python线程方案完整两轮通过，没有再次崩溃。

v2全量归档2658298bytes、11parts，SHA256 fd5b6bdf38869eab2bb61833559ed5549e20c84fb9dc92ca420c032641e4ec11，本地完整核验；实际hook/控制器与源码快照逐字相同。未采集硬件profiler，原benchmark与安装源未改，正式最好仍4K3222.52、16K2440.62及Math102/105。
