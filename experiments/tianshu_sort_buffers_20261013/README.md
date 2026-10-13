# 天数32g 私有索引32服务实验

120项GPU值/稳定int64输出检查已过，M64词表sort micro约+16.6%。真实N64五轮last3 3247.839431 vscontrol3253.050661（-0.160195%），320请求和7greedy通过，不采用。

routes仅M1..7。天数batch>=8走原生Triton top-p，MetaX强制PyTorch排序；M64 sorter micro不是天数大批次热路径。证据及SHA见results/20261013/service_32g/REPORT_32g.md，继续真正热路径位宽审计。
