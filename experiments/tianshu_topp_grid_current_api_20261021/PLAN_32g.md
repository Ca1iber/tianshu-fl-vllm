# 当前407-source上的TopP16K/grid2：扩展C8真实API门槛

当前父profile仍GEO32K/16K、w16/s1、min(B,16)程序，生成源码09156e66...。旧网格在10/19不同profile曾小服务+.807%，未采用，不能当本配置成绩。本目录父prepare固定current prefill INT8候选，原13serve argv/62env，双方同源；仅TopP支持B8..64的几何不变16K/8K tile、程序min(B,2×原grid)、一块64行持久scratch。保留父Router，不假设其.fn，不替换Sampler/alias、参数、RNG。

CPU执行真实外层路由，原对象刻意只支持__getitem__没有.fn；验证B/shape/dtype/device/stride/TopK/块大小边界、参数转发、单scratch与当前父配置一致性。组件不构造模型，不能用它证明模型显存计量或质量。

GPU25场景×6p×2phase=300mask，含11种B8..64、B8/C64 BF16/宽、mask/ties/flat/onefinite、有步长输入；33实际Sampler无/部分/全部逐请求RNG、8次跨8/32/33/64图生命周期、8原生fallback。保持原逐行判据：值不变且（该行与当前基线逐位相同或FP64质量/最小支持/排序1e-6通过），strict false继续记录。相同GEO源码、32步与1e-12区间、p与RNG不变。

每分布保存初始完整input，GPU两阶段保存实际p与两臂packbits有限mask/完整FP32输出SHA。原循环每个p在phase1累计roll17，CPU依pindex+phase精确恢复，重新重建600完整FP32输出SHA并对每行FP64复核。strided输入和真实工作输出均130567行步长。计时只p=.95的五次交替×10热Graph回放，共500event，每次相同输入恢复，不含CPU存盘，不是服务吞吐。

执行前GPU空闲，35分钟组件超时/23:58北京硬停。GPU启动后py/sh/PLAN/CPU_GATE冻结；真修复新目录，元数据恢复另CPU目录且不重跑GPU。完整PT/编译产物仅容器，GitHub代码报告关键结果；用户dirty/Ops不动、无本地副本、无子agent。

API通过后才另新服务目录接current407：同源小4K C64N64每臂3轮排首轮，有稳定收益再新105Math>95及原N256。不借旧Math，不常规16K。shared sampler scratch/临时/模块必须在原生memory_profiling_fl真实with内yield前预热两臂相同，不reset peak，不能用此API组件的显存代替模型native计量。
