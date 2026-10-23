# 原生W4分组首版：group256实际运行时被拒绝

CPU准备已完成：新增group64/256各42层真实Down bank，全部打包逆变换、scale/zero和FP64参考预检通过；原group128文件逐SHA复用。首次GPU源码冻结4e62871。

GPU在预热阶段退出，尚未进入变化输入数值检查或任何计时。实际`aten._weight_int4pack_mm_dotsaddz`调用报错：`Expected qGroupSize == -1 || qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128`。group256不在实际接受集合，不能根据安装库中混合提取的允许256契约字符串声称这个实际入口支持256。该记录修正先前的功能推断，不代表组件有任何速度或模型质量结果。

原始失败`/workspace/benchmark-results/20261023_native_dots_groups_Down64_GPU_tianshu/result.json`及队列日志保留，不覆盖、不改预期。全部冻结源和CPU门禁SHA不变，进程退出后GPU空闲。下一版仅移除该不支持group，复用现有bank/参考比较64与128；原版不做原样重跑。没有修改模型checkpoint、已采用r3或任何正式成绩。
