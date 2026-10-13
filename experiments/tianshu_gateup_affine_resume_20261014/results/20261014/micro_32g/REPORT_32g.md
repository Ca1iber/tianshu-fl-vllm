# 天数 BI-V150 32g：双累加器GateUp/SiLU恢复验证正确但更慢

此前affine版因为half不是constexpr而编译失败，旧WIP已修正但未GPU验证。本次把原样WIP复制到新目录，原文件不编辑、不暂存、不提交；独立8场景×7配置共56项，编译错误0，CPU选定参考/GPU FP32/native固定容差/尾部/Graph全通过，56/56逐位native相同。不能把旧编译失败改写为旧版本曾正确。

基线为真实vendor NT BF16 projection加独立torch.compile SiLU/product；从模型编译缓存中冻结匹配12288/6144的actual activation body，明确projection BF16后SiLU/product用FP32、最后BF16。候选用两个连续weight loads和两个dot accumulators，half显式constexpr，保留这两次舍入；无权重重排、量化或installed修改。

|M|真实I6144小scale最好配置|baseline ms|fused ms|耗时变化%|
|---:|---|---:|---:|---:|
|8|{'bm': 32, 'bn': 128, 'bk': 32, 'warps': 8, 'stages': 1, 'affine': True}|0.094148|0.129069|+37.091932|
|16|{'bm': 32, 'bn': 128, 'bk': 32, 'warps': 8, 'stages': 1, 'affine': True}|0.088247|0.130953|+48.392929|
|32|{'bm': 32, 'bn': 128, 'bk': 32, 'warps': 8, 'stages': 1, 'affine': True}|0.096308|0.121779|+26.448225|
|64|{'bm': 64, 'bn': 128, 'bk': 64, 'warps': 8, 'stages': 1, 'affine': True}|0.102725|0.116719|+13.623347|

主M64最好仍慢13.623347%，M8–32也更慢，不进入服务/Math/full。小I130的对齐和边界用例属于数值验证，不能当模型全I6144收益。全部7轮交替Graph、每轮每图50次原始样本保留并核验；源码和actual cached activation snapshot也保留。

全原始包2531795bytes/10分片，SHA256 8f489624bc16303341112dd7790d75d0a3b6ecf86eaf321504e86da88e10760d，分片/整包、56实际记录、原始timing/CPU和两份源snapshot逐字核验。原件service-results/20261014_tianshu/gateup_affine_resume/20261014_gateup_affine_resume_tianshu。当前正式最好3640.16/728.03/TTFT4465.17/newMath103/tag tianshu-best-decode-kv32-w2-20261014保持。

结论：本次把旧原型推进到可编译且数值正确的checkpoint，但没有性能收益，不采用。硬件profile：TODO有效图内kernel覆盖；后续先核验ixkn-cli剖析覆盖能力，再选新瓶颈。无新16K服务，安装源/权重/原benchmark不变，沐曦只读。
