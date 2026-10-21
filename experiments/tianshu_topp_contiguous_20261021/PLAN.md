# 原生 top-p 连续布局对照（新源版本，旧strided结果不改）

补测实际FP32 row stride130560；旧topP_launch全部stride130567不能代替生产数据。数值JIT未改，仍仅warp2/4/8/16，原生真实wrapper基线、显式4 warp同binary。新work storage为[B+2,V]，使用中间B行，完整首尾行guard；scratch仍独立头尾guard。旧源码、原始结果和逐位不一致判定不变，本新实验不改判旧门槛。

保留原12场景、三次变输入、六种含非有限分布、B8/B64、第三phase多p、180全输出SHA和mask，完整输入/RNG/全部guard。7轮轮换×20replay，每次Graph先恢复完整未过滤logits，baseline和所有候选都有同样copy。正确记录所有native差异，不能宣称任意warp逐位相同。保存原始输入和mask可由独立CPU重建全部输出SHA，并与原生/理想nucleus对照。

当前已知strided组件warp16可减少约28-35%过滤耗时，但p≈.95有32/1208行与原生不同，最大截断TV约.00187；没有新模型成绩，原native bitwise gate仍false。此连续组件仍只是组件证据。若此轮有清晰收益，后续新模型profile明确为允许归约数值差异的探索候选，先4K小A/B、收益明确才严格>95%新Math105与正式4K；不把原组件不等价改为通过，也不忽略原生自身和理想FP64不同的诊断。

源清单含primary组件/native文件，不是全Python传递闭包。GPU最多30分钟且北京时间23:58硬截止，只允许唯一任务。GPU启动后所有py/sh/PLAN/CPU_GATE冻结，真正修复新目录，元数据恢复独立CPU目录。所有原件仅容器，Git只代码/报告/关键JSON/CSV。当前推荐版仍5586.28。
