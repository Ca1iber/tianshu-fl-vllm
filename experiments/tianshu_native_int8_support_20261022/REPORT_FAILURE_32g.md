# V1先在M8 host约束被拒绝

原生torch._int_mm要求self.size(0)>16。第一个M8/K2048/NN在host形状检查处失败，没有完成任何INT32矩阵数值核验，也没有进入Graph。原源码/PLAN/raw failure保持，不能由此推断M64或CoreX整数计算不支持。V2仅将目标限定为M64的K2048/K6144，两种布局和3相位核验不变；M8维持原实现，不偷偷补齐改变计时范围。
