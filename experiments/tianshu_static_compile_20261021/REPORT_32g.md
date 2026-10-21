# C8/C64原生专用编译：小4K+.254%，未建立稳定收益

同发布407源prefillINT8基线、4096预算/seq64/async、原13serve参数及62环境；本实验冻结420源。原生compile_sizes默认[]→[8,64]在实际EngineArgs deepcopy前修改，调用后恢复原参数对象，显式配置及不支持路径保留。CPU8400实际Range/dispatchAST及原生配置copy/归一化等检查先通过。

|臂/轮次|total tok/s|TTFT ms|新增autotune|
|---|---:|---:|---:|
|control1，排除|5319.28|12383.75|0|
|control2|5512.35|10391.02|0|
|control3|5500.34|10398.17|0|
|candidate1，排除|5330.33|12383.76|0|
|candidate2|5527.66|10401.64|0|
|candidate3|5513.01|10376.54|0|

排首轮均值5506.349497→5520.334938tok/s，+0.253988%；TTFT10394.594691→10389.085657ms，-.052999%。两臂吞吐极差.218111%/.265308%，候选极差超过此次均值增幅，未建立稳定模型收益；未采用、没有新Math/N256。当前推荐5586.28/3780.63/Math102保持。384/384timedrequests usage/成功、独立原始重算、零抢占/FutureKV释放和真实C8/C64路径均通过。

实际native configs/returnedRangeEntry证明candidate捕获时为[8,8]/[64,64]，control[1,4096]；所有piece与真实父Graph replay计数相连，C8 canary及C64至少100次。不是仅根据配置、Python分支次数或编译日志声称命中。参数/banks/算子精度路由相同，prefillINT8实际M128/2048/4096，两臂24INT8warm、43bank同payload1,326,938,112B。

额外专用编译增加启动成本；native日志报告C8/C64range生成，当前完整端到端启动还含加载/Graph/预热，不能把这些单条计时机械相加当启动总差。实际KVcontrol40731、candidate40143，少588块；可用KVbytes少303108608B，memorycallback时两臂torch_memory_allocated/arena完全相同。原因尚未归属activation coldpeak/non-torch/Graph estimate，不因推测调整预算。详细native源码审查见../tianshu_reservation_20261021/STATIC_MEMORY_PROFILE_REVIEW_20261021.md。

原始根/workspace/benchmark-results/20261021_static_compile_service_{control,candidate}_tianshu。SCREEN_GATE/SCREEN_ROUNDS/KEY_ROUNDS在GitHub，完整回答/profile/源码快照仅容器。新KV bound425源框架候选在独立目录，从原407profile开始、保留8192arena，不与本static方案叠加，避免把未测组合归到本结果。
