# M范围首个组件：QKV24点完成，原计划在O/M8原生对照失败

三角色24M×2相位原计划144项，实际51项完成：QKV24点两相位48项，O的M1两相位与M8第0相位3项。随后O/M8第1相位原生对照对FP32参考有12元素超原.02+.01阈值，max_abs .044921875；matched NT、packed-W、packed-AW该相位全部通过，最大误差.00048828125。原GPU complete=false/phase=failed保留，Down未执行，不能改判整轮通过。

独立CPU重读已完成51项保存的真实选行列向量，逐项F64重算通过，所有已完成项的完整GPU在线输出检查也全通过。GPU只保存选行列与完整输出SHA，不存在完整输出CPU重载的证据。PARTIAL_GATE明确区分部分复核成功和原计划失败，全部源/启动快照未改。

QKV显示明显M分化：同一套BM128/BN128/BK32/W8中，M384 native=.09304ms、同tile NT=.06077、packed-W=.05700；M1152 native=.26368、同tile NT=.16603、packed-W=.16791，主要是tile/配置收益而非W重排。M576 native=.16064、同tile NT=.24853、packed-W=.24757，迁移反而慢；M4096 native=.49323、packed-W=.56296，也慢。输入A再重排大多增加开销。它支持按role/M选择，但还不足以认定整段范围或模型收益；更小BM处理非128整倍数是可测试的新配置。

原生公共入口已实查确实是_iluvatar.ops.mm，源SHA4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090，非误用通用NVIDIA实现。小M数值差异原因尚未确认；下一步新目录诊断保存完整O8输入/W/输出、实际编译常量、eager与Graph及权重变化前后，先区分原生split累加、缓存或图语义，再决定后续对照，不提高阈值。

没有新服务或Math，不替换已正式通过450-source 4K5618.01/TTFT3835.46。原始目录/workspace/benchmark-results/20261021_m_range_BF16_GPU_tianshu，失败fixture有选行列保存，但没有原失败完整12元素的完整输出张量，因此新诊断不可声称逐位重放了原失败输出。
