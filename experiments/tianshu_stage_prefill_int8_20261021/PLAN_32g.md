# 当前正式407源/62环境项版本的Graph外阶段诊断

正式最好5586.28与Math102/105不变。上一阶段分析是10-20的205源版本，当前已采用NN GateUp、选择性LM精度和prefill INT8，因此旧88.22% forward比例不当作当前结果。

复用旧验证过的Graph外事件观察器，仅替换父hook为当前prefill候选、记录/arm路径从现有DENSE目录推导、取消额外测量环境变量。原始serve argv、全部62环境项保留，唯一PYTHONPATH指向新观察入口后链到原完整hook。原407源必须与正式RUNTIME_BINDING_GATE逐一同hash；不改算子、精度、调度、采样参数、KV格式或数学路径。

先128/16 seeded777 canary；64并发4K/512暖一批，再同规模测量一批；后canary必须逐字一致。测量批连续32步C64纯decode后，在40步FULL/padded64记录forward、logits、sample、total外部事件及CPU wall/threadCPU。事件预分配，窗口边界两次同步，禁止在捕获中插事件。原验证器保留全部模型/42GateUp与LM/pairedK/KV/原RNG/零preempt/元数据释放/原argv检查，并加零ready后autotune。

阶段stream区间包含host提交空隙，不是单kernel活跃时间，不能相加CPU/GPU或按本次响应算正式吞吐。无新候选，不跑Math或正式4K/16K。CPU先验证原hook/driver的严格逆变换、62env与407源子集、字节快照与原argv、所有新源码语法、torch.cuda未初始化。GPU前冻结全部代码/PLAN/CPU_GATE；独立CPU后处理重算40帧区间汇总和128响应usage、前后canary，校验原运行时证据。仅一次运行，2400秒超时和23:58硬停。失败保留日志；后处理错误只做独立CPU恢复，不重复GPU。

所有原件只在天数容器；GitHub保存代码、报告、关键门控及40帧小结果，不上传大日志/张量，本地不留副本。

CPU fixture初次最后的唯一ModelRunner路径断言失败：正式407源中同时明确列出仓库与site-packages两个文件，它们均为相同b0144哈希。实际内容与正式版完全一致；v2改为精确检查这两个路径集合及两者hash，不是忽略路径或放松源码检查。GPU尚未运行，第一次日志和CPU快照保留，新CPU快照目录单独v2。
