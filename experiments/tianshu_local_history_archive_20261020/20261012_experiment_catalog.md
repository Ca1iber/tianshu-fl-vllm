# 天数实验目录（2026-10-12 12:09）

当前已完整验证的官方4K最佳为2712.55 total tok/s、542.51 output tok/s、TTFT4720.38ms，对应独立Math102/105（97.14%），零请求错误。四轮2127.87/2587.71/2751.28/2798.67，排首轮后均值2712.55；后3轮极差/均值7.777174%。最佳标签tianshu-best-native-prefill-20261012指向aa8fac4，本地和容器一致。原先2504.08配置作为历史最佳保留。

预约已延至2026-10-13 00:00（北京时间）；23:58停止GPU工作。全部实验仅在天数自有仓库，沐曦参考仓库只读；未推送GitHub，未运行优化版完整16K。

|实验|负载与结果|当前结论|完整证据目录|
|---|---|---|---|
|Hybrid native Decode / FA2 Prefill|官方4K2286.09、TTFT7569.57；Math100/105|先前正式最佳保留|20261011_hybrid_official4k_tianshu|
|adaptive Decode|数值/CPU/scratch80项及联合13项通过；独立数学102/105；较短N64吞吐+3.2322%|后续正式组合包含该项，未把组合收益归因到单项|20261012_adaptive_decode_tianshu|
|预算8192|标准较短4K/1K+3.7967%，Math102/105；官方2272.25，比2286.09低0.6054%，TTFT变差17.6830%|性能负例，质量通过也不采用|20261012_budget8192_official4k_tianshu|
|FA2 tile128|42项数值及24项CPU通过；C64/N128、4K/1K+9.4847%；正式组合2504.08，Math103/105|历史正式最佳保留|20261012_fa2_tiles_tianshu 与20261012_fa2_tile128_official4k_tianshu|
|tile128下预算4096|C64/N192、4K/128：6014.18→5743.14（−4.5068%）|不采用；新Native配置的预算需要另行隔离验证|20261012_fa2_budget4096_tianshu|
|固定gather|100项整数逐位与尾部检查通过；旧profile服务稳态−0.0336%|不采用；启动缩短单独记录|20261012_gather_scope_tianshu|
|metadata整数融合|15/15逐位与尾部检查；服务6020.0508→6020.0243（−0.000440%）|算子微测快不代表服务有收益，不采用|20261012_metadata_fusion_tianshu|
|采样融合|105/105固定噪声CPU参考检查；C64/N192、4K/128：6009.90→6022.94（+0.2171%）|收益不足，不采用；没有新Math或正式测试|20261012_sampler_service_tianshu|
|M2048 GEMM|44/48数值通过，4个共享内存编译失败；短服务+1.2162%|标准4K/1K +0.5007%，TTFT−1.6399%，收益小，未采用|20261012_prefill_gemm_tianshu|
|M8 split-K GEMM|120/120数值/CPU通过；初测名义+31.5081%，对照计分轮出现10项调优|标准C64：2534.99→2533.34（−0.0651%），无新增调优；不采用。充分预热C8仍待确认|20261012_small_gemm_tianshu|
|原生扩大Prefill tile初版|30个场景18通过、12失败；零KV prefix首query出现半幅输出|不能采用；失败源码和数据保留|20261012_native_prefill_tianshu|
|原生Prefill显式L0|修复后30/30场景及24/24CPU通过；C64/N128、4K/1K：2534.99→2742.83（+8.1990%），TTFT−26.5135%；独立Math102/105|正式4K2712.55/4720.38，采用为最佳；完整备份已核验|20261012_native_prefill_tianshu|
|直接Decode块映射|只改变一查询序列的块映射，算术不变；计划10个Graph/CPU/尾部场景|待数值筛选，不能声称收益|远程排队目录20261012_decode_mapping_tianshu|
|GateUp与SiLU融合|保持权重布局和已编译模型的舍入边界；8场景×7配置待运行|新原型，尚无GPU数值或收益结论|远程排队目录20261012_gateup_silu_tianshu|

现有证据已逐块和整包SHA校验；当前新正式结果已完整记录，后续实验完成后继续补充。初测、短测、算子事件计时与正式N256四轮均值分别记录，只有对应独立Math严格>95%的配置才考虑正式采用。

调用链补充：Worker先安装adaptive，再安装Hybrid，混合Decode prefix传回的original已包含adaptive包装器。不能误称全部混合prefix固定S64。原始Decode联合tile扫描基线S16/TILE16，扫描大tile没有包含S4/TILE16；各自结论按具体实验引用。

已预热NoGraph C16的旧profile中，记录到的GPU时间MM39.15%、FA233.55%、native3D17.96%。这些比例仅对该诊断有效；新原生配置的C64预热profile已排队，不能直接拿旧C16比例解释正式C64 Graph benchmark。

原生Decode warp/stage筛选已准备：当前adaptive分段、原始M16/BLOCK_Q2网格与算术，8场景×14候选加原始控制；GPU尚未开始，等待融合数值筛选清理后运行。控制的Triton默认stage不作假设，记录实际metadata。
