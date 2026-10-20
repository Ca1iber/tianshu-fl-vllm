# 受控复现Philox跨界冷停顿，四类入口预热后消除本样例调参

实际LibEntry.key对N、seed和offset这些do_not_specialize参数保存类型，不保存整数值。该模型B1..64的N全部属于i32；原生Philox从有符号int64状态字读取seed/offset，各自可能落入i32/i64，所以只需四种入口组合。CPU直接调用原生key函数3584次验证64批及signed-int64边界，实际四类入口，N值共享key。第三轮B37的冷调参触发条件是offset越过2^31；B37为当时真实批次。

## GPU受控冷态证据

隔离进程只把目标原生tuner的ConfigCache与BenchmarkCache换为空内存夹具，保持NORMAL模式、原4候选和原25/100协议；不删除或改写用户SQLite记录。offset2^31的B37首个冷调用实测2.118073s，测了4配置、benchmark命中0。时间包含首次调用所需的查找、JIT及调参开销，不当成纯GPU计算时间。

保留已选低位/default入口，V2用私有CUDA generator在代表B64上预热seed i32/i64 × offset i32/i64四类入口，检查实际cache内存在相应key，所有预先存在的entry对象保持。该实验阶段四调用耗时0.457947s；此时B37基线已经建立部分编译/配置缓存，不能当作完全空环境的生产启动总成本。

之后跨界B37调用0.000406s（0.406ms），日志无新调参；四类入口×B1..64共256次正常API复测也无新调参。实际4个入口key、2个ConfigCache key（B37/B64）；这是按入口类型覆盖，并非构造64个配置条目。默认CPU/GPU RNG逐字不变，低位B37的4830720个FP32输出逐字保持；本次高位样例的差异字数同样为0。

## 配置和证据边界

本次内存重调的B37/B64都选BLOCK256、4warps、3stages；用户原数据库B37为256/4/3、B64为512/8/3，实验后逐项保持原值，原对象与入口缓存清理均恢复。输出一致结论限定本次实际配置；部署走原数据库512路径时不能借用256配置的逐字结论，若接入模型仍需真实服务与必要的新Math。

尚未接入服务、未采用、没有新Math/N256成绩。原始根目录：`/workspace/benchmark-results/20261021_sampler_exp_cold_v2_GPU_tianshu`。约77MB原始tensor和日志只留容器；GPU_GATE、CACHE_EVENTS、代码和本报告进GitHub。
