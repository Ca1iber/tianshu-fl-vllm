# r3 + 单一Down64 W4质量控制器

使用service/PLAN中的唯一精度变更；原r3 903个源必须完整保留；新增16源共916模型源。原始serve13 argv保持、63环境键仅入口改变（C64沿用既有FutureKV control）。所有控制器及其依赖在第一次GPU前冻结。

C64完整105题先行，采用官方temperature1/top_p.95/max_tokens32768/timeouts和本地Level3数据，错误、空输出、cap按原严格审计计入总数，>95%即至少100/105。通过再同版本C8；准确率未过不重跑同版本追分。不得自动启动正式benchmark或16K。先确认 GPU空闲、端口9031空闲、原冻结缓存完整；截止2026-10-23T15:58Z。

原nativeDown BF16 M64捕获检查被显式替换（新复制审计要求原route没有M64 capture，新独立审计必须42个实际W4 capture、实际worker内存探测42scratch探针、原checkpoint/GPU weight/pack/scale/zero全SHA/库SHA/所有真实model参数buffer storage计入）。调用原875+ GU/KV/attention/scheduler/sampling/LM全审计、原Down的其余每一条、原84个attention M64 BF16审计。旧来源、record不改写，无代理记录，无旧BF16/INT8调用冒充。链接原C64真实M64模型图执行证据与新W4捕获，Python调用数不当作重放数。

代码、报告和关键评测结果推天数GitHub，全部原始数组/日志/缓存保留容器，不在用户本地存项目文件。完整模型质量合格后另冻结新的短4K控制器；本脚本只允许Math。
