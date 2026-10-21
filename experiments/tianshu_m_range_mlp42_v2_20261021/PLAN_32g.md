# 42-bank MLP v2：实时stream与动态Graph/eager证明

独立stream诊断三相位全复现：cached handle=0，capture stream非0；fixed graph输出全保持毒值17，实时getCurrentRawStream Graph全输出与eager位相同且CPU FP64通过。旧MLP所有性能结论已撤回，旧数据及源保留。v2每次activation实时取raw stream；每M为全部42bank各捕获control/candidate图，三输入相位先毒化activation再replay，完整输出与eager严格bitwise、Down阈值通过；整个42-bank计时图再毒化activation/output，独立检查最后bank同eager，确认图内实际计算激活，之后7组交替3次Graph。其余量化/真实42权重/24M/CPU选点/阈值均原样。原组件流依旧非transformer，不作正式成绩。60分钟或23:58北京硬停，源GPU前冻结。
