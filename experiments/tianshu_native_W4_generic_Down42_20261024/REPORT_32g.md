# 标准预打包W4：全部数值通过，真实Down累计耗时约60.3倍，不采用

2026-10-24，BI-V150，GPU PID31828正常complete。比较相同42个当前r5 Down量化矩阵、M64/K6144/N2048/G128：dotsaddz自然pack与标准torch generic预打包。逐文件匹配r5载荷清单，CPU将全42矩阵解码为BF16逐元素一致，没有重调量化参数或修改原checkpoint。标准converter inner_k_tiles8仅在启动执行。

|42真实bank前向热Graph|9轮中位ms|极差/中位数|
|---|---:|---:|
|当前dotsaddz|1.971602058|1.73019%|
|标准generic预打包|118.951416016|0.04493%|

九轮正反交替×20重放，计时不含converter，无独立CPU准备/哈希/归档/推送。候选约60.332倍耗时。控制组差幅超过1%，不能据此比较其它日期1.944ms等历史小差异；不影响拒绝本次数量级更慢的候选。不重跑挑轮，不接模型、Math或正式benchmark。该负结论只覆盖本标准backend/inner_k_tiles8/shape，不声称所有预打包实现都慢。未profile原生内部kernel，不猜测软件模拟或硬件原因。

三组真实GU激活fixture、每层独立分配且按layer滚动行，并非42层各自捕获激活。全42的eager/repeat/同Graph改变输入及NaN毒化重放均有限、各自逐字相同；全部输入/原pack/params/原始high-even字节/原生converter产物的内容与version保持，带边界的输入guard和CPU/GPU RNG通过。初始协议允许既定数值门槛内的重复舍入差异，但本次没有观察到这种差异。

保存33030144个全42 Graph输出字；层0/20/41三相位、两backend、eager/repeat/Graph共7077888个输出对原独立FP64矩阵逐项通过，未将其余39层当作全FP64核验。原BF16门槛.012abs+.006referenceRMS、相对L2≤.006未改。CPU另复核全部源/快照/3份SDK与checkpoint SHA、物理roundtrip、42条真实converter shape/dtype/stride/SHA、原始九轮event/顺序/中位数。结果见GPU_CPU_GATE。

额外native pack+params共280756224B=267.75MiB；原载荷保留。测试另保留high-even输入用于完整不变性检查，生产版本可在startup转换后释放；此处没有生产模型memoryprofile资格，也不宣称净显存节省。原生opaque预打包数组只保存metadata/SHA，按原42载荷和固定SDK可重新生成，不重复持久化大数组。

完整输出、选定eager/repeat样本、CPU来源副本、converter元数据、原event和日志保存在容器SHM及持久压缩档，tar逐文件compare通过；原42权重文件及9份独立FP64参考仍在CPU_GATE指向的持久父路径，未重复复制。归档SHA/位置/大小见RAW_ARCHIVE。代码、报告、关键CSV/GATE推天数GitHub，用户本地无副本。正式4K最好仍r5 6123.31total/1224.66output tok/s、TTFT3202.70ms。
