# 天数BI-V150 32g：M4096 GEMM tile/group

独立设备微测，非模型服务吞吐、非官方分数；不采用到当前最佳4000.99。当前完整服务相同S8、KV16/w1、page16基线，改变Graph输入检查保留。未采硬件性能计数profile。

40候选全部通过，四投影各10组tile/group；560event重算，3份源快照核验，改变Graph输入的GPU FP32和CPU样本检查通过。

|投影|最优候选耗时变化%|配置|
|---|---:|---|
|qkv|+7.797461|{'bm': 256, 'bn': 256, 'bk': 32, 'splits': 1, 'warps': 16, 'stages': 2, 'group_m': 1}|
|o|+6.734953|{'bm': 256, 'bn': 256, 'bk': 32, 'splits': 1, 'warps': 16, 'stages': 2, 'group_m': 1}|
|gateup|-2.828071|{'bm': 256, 'bn': 256, 'bk': 32, 'splits': 1, 'warps': 16, 'stages': 2, 'group_m': 8}|
|down|+14.793037|{'bm': 256, 'bn': 256, 'bk': 32, 'splits': 1, 'warps': 16, 'stages': 2, 'group_m': 1}|

只有gate/up约2.8281%减时，QKV/o/down即使最优候选仍分别慢7.7975%/6.7350%/14.7930%。不把M2048旧结论迁移到M4096，不接入其它三投影；gate/up单点暂排在.cv之后。

原始归档3084073bytes、12分片，SHA256 `56283d5833811b5dddd9b175788fbe3a8f7ec8dff375fc8426c8063415621146`，分片及整包验证通过。完整行见RAW_AUDIT_32g.json。
