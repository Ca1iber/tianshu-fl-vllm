# KV prefill范围收紧：实际生效，小4K+.0543%，未采用

V3原N64/C64、4K输入/1K输出、每臂3轮排首轮，426运行来源/62环境、原13serveargv，基于发布407源4096预算，arena固定8192pages，无staticcompile叠加。384/384成功，原usage/seed-canary/源码/真实C8/C64Graph/42层及43bank审计保持，全部6轮无autotune、零抢占/FutureKV全部释放。独立verify_pair完成。

|臂/轮次|total tok/s|TTFT ms|
|---|---:|---:|
|control1，排除|5316.17|12408.90|
|control2|5509.21|10390.77|
|control3|5505.55|10378.12|
|candidate1，排除|5328.50|12337.07|
|candidate2|5515.03|10352.15|
|candidate3|5505.72|10394.62|

均值5507.383910→5510.374449tok/s，+.054301%；TTFT10384.443103→10373.387328ms，-.106465%。极差.066458%/.168823%，均大于增幅，未建立稳定吞吐收益，没有新Math/N256，当前5586.28/3780.63/Math102推荐不变。

实际active stage记录证明已收紧：63decode+1prefill时globalmaxK4160→prefill4096、pages260→256；62decode+2prefill时4159→4096、520→512；C8canary某混合阶段129→128、63→56。没有改变全局decode参数，原Route与GPUcu保留，完整key含前缀。4K固定输入场景的实际暂存减少约1.5%，与组件刻意构造短prefill/长decode的14倍/26倍gather情况不同，不能外推。计数是已保存Python观测下界，不是Graph replay或每请求调用总数。

CPU helper2048元数据/119960逻辑页、policy12项native Route AST和GPU24场景1.55亿有效元素全部通过。已有小arena使用CapacityBoundedDensePrefill子类，V1错误要求原始类名而启动失败，V2又因未注册动态模块的class inspect失败，均无GPU模型评测。V3实际复现未注册loader，用方法co_filename和继承identity核验两份源，完整CPU reserve1.34亿元素/shape/reservation相同，拒绝错容量/未包装类。V1/V2代码/门禁/失败原件保持，V3为独立新版本。

原生预算仅观测不修改：control KV40731，candidate40723，差8块。两臂weights6429220352、peakactivation402702336、Graphestimate0（flagtrue）、实时Torchallocated6691412480相同；native non-torch1352663040→1356857344差4194304B，解释实际KV可用bytes差4MiB，不能归因于模型精度或arena增加。extra packed payload1326938112B与原发布相同。没有新增profile_run/reset_peak/config修改。静态编译先前少588块的原因仍待独立观察，不能用本不同profile直接归因。

原始根/workspace/benchmark-results/20261021_prefill_key_bound_service_v3_{control,candidate}_tianshu。SCREEN_GATE/ROUNDS/KEY_ROUNDS为关键结果，完整回答/profile/快照留容器。下一项独立native_memory_probe仅做startup+8个C8canary记录预算分项，不声称吞吐/质量，也不改KV预算。
