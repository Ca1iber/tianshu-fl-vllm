# 后续方案：以当前407源为对照，避免重复旧profile

QKV9×20已独立CPU复核，M4096 .492038→.428116ms；小M8/64/129负收益。O独立诊断9×20已GPU完成，CPU复核正在执行。旧双投影父运行O8/index4/phase2失败保持failed，不能覆盖旧源或Gate。新O逐项保存原BF16误差谓词false，不放宽，候选INT32/量化输出严格；32次同一O8原phase0输入完整输出相同/阈值全过，但后续3项selected原BF16超差，根因未定。

TopP16K/grid2并未在当前best开启。旧tianshu_topp_grid_api_v4_20261019/SERVICE_REPORT_32g.md在旧S8/2048-budget组合曾稳定+.80713%，API156mask/21RNG/6生命周期/8fallback过，未Math/正式。当前仍GEO32K/16grid（09156e66），可以新目录复测当前407父栈与扩到B8..64，保证C8 Math实际覆盖新数值路径；这不是今天重复warp16。原grid_route正确包装已存在Router，不假设original.fn，候选直接用GEO JIT且fallback全委派，kernel_route自有64×130560 FP32 shared scratch，仅顺序stream。

建议下一步先新API CPU/GPU门槛：原B32..64扩B8/16/24/31，加B8 BF16/宽/mask/ties/flat/onefinite；原每行值不变且（等基线或严格1e-6 mass/minimality/rank）判据不放宽，保留strict false。Sampler/RNG/alias不替换。生命周期扩8/32/33/64交错8次，RNG33例，fallback改1/2/4/7及原TopK/no-filter/65。父profile必须current tianshu_prefill_int8_service.prepare(root,'candidate')，新模式可由hooks_control/candidate取得、记录路径从原dense目录parent推导，保留62env和原13argv；运行源数CPU实际算。API组件与后续service另目录，GPU启动即冻结py/sh/CPU_GATE。

若保存完整证据，勿重复276组全矩阵17GB：按distribution存一次input，旧probe每个p的phase1累计roll17，故按pindex+phase恢复；每组存实际p、两臂packbits finite mask、完整FP32输出SHA。GPU先查保留值不变，CPU用input+mask重建全输出SHA并FP64重算mass/minimality/rank，raw仅容器。

潜在模型组合：按O4096完整结果决定QKV/O是否接入，仅aligned warm M>=128，其余真实原路回退；原GUP/NN/LM/K8V16/FutureKV保持。额外42层payload QKV220631040B/O176504832B，原BF16保留，两臂相同pack。process_weights_after_loading中真实DeviceMemoryProfiler计费、前缀/shape/stride/hash/version原样核验。临时/模块/sampler scratch在原memory_profiling_fl上下文内yield给原profile_run之前预热两臂相同，RNG不变，不resetpeak；新增QKV量化应在profile_run启用以正确计入联合峰值。native峰值在with内读取，yield之后warm无效。decode捕获8/64保持fallback。parent audit始终用candidate保留407全部检查，新审计另标新模式/42层actual ready路径，C8必须实际走新算子。先N64小4K3轮配对排首轮，稳定收益才新105 Math>95与正式N256，不常规16K。

Goal active，23:58北京/15:58Z硬停，预约到10/22零点。SSHmaster04:27建立16:27前刷新。用户dirty2文件/Ops4cdc3b7不动，禁止本地副本和子agent，每完成服务轮报告。上次长保存命令审批超时未执行；本短记录已实际保存。
