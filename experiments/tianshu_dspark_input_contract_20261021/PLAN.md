# DSpark 七槽输入契约及最大长度诊断

目的：直接执行已安装 DFlash + 旧 DSparkProposer.set_inputs_first_pass 的实际 host/kernel，不加载模型，验证与训练一致的 anchor-first 七查询槽、按拒绝数量扣除后的起始位置、随机跨页 slot、context/query 分离和采样行。10 种 fixture，每种两次改变 past/bonus/page，四种提案数 1/2/3/7、两种 sample view，共160调用。包含上下文1/17/256/257/513、多block、B64、拒绝和接近模型最大长度；GPU输入始终合法，所有query position小于131072。不在真实越界位置执行GPU。

独立CPU oracle逐值复核全部位置、token、slot、seq_len和采样索引。GPU保留输入/输出guard、RNG、hidden identity及原始tensor，结束后CPU再次读取原tensor重算；全部primary源码和CPU gate冻结快照。此处只验证输入准备，不验证真实KV写入、模型、草稿分布、吞吐或Math，不据此更换推荐版。

新policy仅process-scope包装ModelRunnerFL._input_fits_in_drafter：Qwen3 DSpark两种架构按实际固定7槽判断，其他模型维持native规则。CPU执行安装版原方法作对照，并验证辅助层+1实际方法。policy在本GPU输入诊断中不安装到服务，因为没有服务；实际服务接入必须在之后新profile中完成。

所有py/sh/PLAN/primary依赖和CPU gate在GPU启动后冻结，不改、不补文件；元数据恢复在新CPU目录处理；真正代码变更用新实验目录。时间最多15分钟，并受北京时间23:58预约硬截止。只允许唯一GPU任务，输出仅容器、GitHub只推代码/报告/key JSON。
