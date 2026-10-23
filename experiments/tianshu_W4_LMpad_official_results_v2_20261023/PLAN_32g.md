# LM全小批次W4正式4K独立CPU复核

首次正式GPU前冻结，当前Math和短测门槛尚未通过，因此仅预备不执行。必须同944-source候选和完整C8/C64独立质量、同期短4K通过；原始N256/C64/4K1K四轮排首轮，源/controller当前与快照逐SHA、真实新LM8/64 W4/Down42捕获、保留GU/KV/attention/scheduler/sampling与原生峰值、1024成功请求和总token统计、ready后无autotune/0preempt、checkpoint与原argv/env均复核。

按官方脚本保存的数据范围重算raw CSV四轮及summary热三轮的吞吐和TTFT汇总，不声称拥有官方脚本未保存的逐请求TTFT/usage trace；N64短测另有逐请求记录。重放冻结完整runtime及comparison审计，与原运行字段全等。采用门槛固定vsr3 5950.75 >.5%、热spread<1%、TTFT<=3174.58*1.01；历史正式对比不冒充同期AB，同期证据来自独立短测。

任何失败如实保存，不原样重跑挑成绩；只有本门禁complete且promotion_passed才构建新发布，不能以GPU退出码替代完整资格。代码/关键JSON/CSV/报告进天数GitHub，完整原始仅容器；本脚本无GPU导入、不运行模型或benchmark、不修改原始文件。候选模型和本预备验证器不等于已采用成果，16K不跑。
