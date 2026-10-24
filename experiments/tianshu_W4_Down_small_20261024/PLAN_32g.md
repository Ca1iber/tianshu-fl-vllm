# r4 Down 小批次 W4 组件筛选

2026-10-24，BI-V15032G。新目标量化探索。r4 Down仅M64用native W4，M1..63仍执行原native_down::linear的BF16 vendor mm fallback（prefill native INT8仅128/512/4096）。LM已用W4全M1..64，GU W4昨日比融合W8A8慢约40%，均不重复。先考察Down M1/2/4/8/16/32/48/63能否复用r4既有42份G128 W4 payload；不新增权重格式、不改checkpoint、不做训练或微调。

控制直接调用该fallback实际Iluvatar module.mm(x,weight.T)，保持r4源与launch hook。候选M1/2/32直接native dotsaddz，其余零补到32/64，完整分配、padding、native输出切片都计时。采用42个真实不同bank及各自输入计时，避免只测一份可缓存权重；同一组Graphs变3相位X，基线与候选同样原生调用，没有预先缓存结果。

复用原已独立验证的42层量化及L0/L20/L41三相位M64完整FP64参考，重新核对来源/文件/checkpoint全SHA。新输入是这些矩阵的前M行，CPU核对每个完整输出prefix；阈值仍逐格abs(err)<=.012abs(ref)+.006RMS(ref)，relL2<=.006，不因M改变而放宽。仅3层有独立全FP64参考，不声称42层全FP64。全部42层检查完整字节、版本、guards、RNG、重复eager及poison后Graph逐位；先保存输出与失败再判定。控制若不合格，该M不宣称速度收益。单M/单方案失败不重跑挑分。

7轮交替，graph/eager各10次42-bank调用，首轮预先剔除，报告所有M及所有原始时间。计时期间不并行CPU准备/打包。若出现数值合格的明确阳性，再覆盖整个相关M范围并考虑新C8/C64 Math和N64短4K；只有模型阳性才跑官方N256。组件收益不等于模型成绩，本轮不跑16K。实际模型/r4代码不改。原始数组只在容器SHM，结束归档持久目录，Git仅代码/报告/关键JSON/CSV；不在WSL保留副本。GPU alarm1800秒，硬截止2026-10-24 23:58北京时间。
