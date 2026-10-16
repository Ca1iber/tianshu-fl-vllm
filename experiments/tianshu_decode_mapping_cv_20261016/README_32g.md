# 现最佳.cv attention直接序列映射验证

不是新发现：Oct12已有直接映射微测，B64原生attention约-1.3%耗时，但因转DSpark未做服务。当前已换成KV16/w1/S8/.cv，不能直接搬用旧收益；新目录从8deb896缓存源码生成两处可逆修改，仍保留attention循环、mask/softmax、int64地址、reducer与分段完全相同。纯3D单query路径pid0直接序列，网格第0维由q.shape//2+num_seqs改为num_seqs；原映射有空洞，不能只截短旧网格。

先在现最佳环境做独立组件微测，包含1/8/24/32/64、4K/5K、strided Q/KV、随机页表、ragged与0/1内部空query。改变Q/K/V及含空query的cu排列后，清空输出并重放同一Graph；自身FP32+CPU参考、原生逐位比较及尾部哨兵，7交替组各20 replay。没有服务/Math/正式成绩，不同时运行GPU；明确收益后才考虑原服务3+3。候选只改producer映射，reducer仍通用二分查找以支持任意0/1查询，不能假设query索引等于seq索引。

已完成并下载核验64源/112事件/16阶段逐位等价。当前B64满批4K、5K反而慢7.65%/8.04%，不接入服务；B24和人工空query的组件收益另记，不作服务推断。详见results/20261016/micro_32g。
