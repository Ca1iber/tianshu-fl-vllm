# 天数32g CPU冷停顿诊断

已使用Python线程sys._current_frames完成两轮N64/C64/4K1K独立诊断。低速尾部119样本中116在SQLAlchemy、96经过FlagGems build_sql_model_by_db全库反射；高速主要等GPU输出。根因热点已定位，修复A/B尚未实施。报告和原始SHA：results/20261013/diagnostic_32g/REPORT_32g.md。

无ptrace权限，py-spy未取得栈；v1 C异步faulthandler导入期段错误，失败证据保留，当前v2线程方案两轮完整通过。诊断文件只记录代码file/function/line，不读locals，明确不作为性能成绩。source/hook/JSONL栈全部归档。正式模型/安装源/benchmark逻辑未改。
