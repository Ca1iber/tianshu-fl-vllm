# 天数TopP实际最大值归一化

4个fixed候选的10输入严格通过；原生/identity保留已知严格语义错误。32K普通B64约2.93ms vs原生4.59ms。改变输入Graph和CPU FP64 1e-6门槛未放宽。尚无模型成绩，详见[报告](results/20261016/micro_32g/REPORT_32g.md)。
