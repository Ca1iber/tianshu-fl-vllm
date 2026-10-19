# K8V16 CUDA Graph实际分段诊断

复用当前4997.05已验证profile，保持算法/数据类型/预算/参数/RNG，增加只读记录钩子。
43项实际环境=原41项（PYTHONPATH转为组合hook）+诊断开关和记录目录。
跟踪实际CUDAGraphWrapper对象/entry/graph identity，关联捕获时42层真实Producer配置，
以及运行时重放使用的batch descriptor与CPU已有的computed token长度。
图捕获分段不能只凭启动日志或producer中出现过某个配置就推断，必须完成实际连接。

它不改任何attention参数、返回值或采样RNG，不在图内加入GPU timing/event/copy。
记录和文件IO会影响主机耗时，本诊断不是性能分数或新benchmark成绩。
使用原serve argv，六组C1/8/16/24/32/64，4096输入/128输出，共145负载请求，
另3个seeded canary需与既有K8V16参考文本完全相同。
沿用原KV布局/内存预算审核，仅把实际环境数量检查扩为43，原41项参数保持绑定。

CPU直接执行实际安装cuda_graph.py的__call__ AST，覆盖capture/replay/NONE/无context/
PIECEWISE接受/异常转发，检查原参数/返回值/异常identity和图到捕获配置的关联。
GPU诊断本身仍待验证，不提前声称当前短/长上下文分段存在问题。

V1负载和canary完成，但其普通CUDAGraphWrapper未走实际模型路径，关联审核失败，原件保留。
V2明确绑定722683b4的Breakable包装器，区分wrapper capture/replay与实际stream capture，
记录每个capture对象的GPU graph/eager片段、实际GPU graph identity及eager闭包函数名。
不把绕过普通包装器误报成没有使用CUDA Graph。
