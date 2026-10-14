# 天数32g：Decode CTA三轴排列实验

新的独立算子候选，尚未执行GPU，不采用、不借用其他配置的Math/官方分数。基于实际已验证的.cv生成源8deb（L0、guarded KV16/w1、B64S8），只改变q-block、KV-head、segment逻辑轴与physical program_id轴的对应关系，并同步置换launch grid。保留全部原逻辑CTA和无效query CTA，没有使用此前负收益的直接序列映射，也不改算术、分段、整数宽度、tile、warp或reducer；2D使用原0/1映射。

CPU对6个排列×8个batch×2个query填充模式共96种grid、37488坐标逐项枚举，证明逻辑CTA恰好一次且无缺失；6份生成源均可将三条program_id赋值逆替换，逐字恢复原.cv源。CPU不是GPU数值或性能证据。

probe_grid.py准备17个场景（4K/5K、strided K/V、0/1 query、tile边界），比较实际原.cv与6个生成副本（含identity控制）；每种保留FP32/CPU参考、原输入逐位对比、改变query和K/V后的Graph重放与逐位对比、output/scratch尾部、9轮交替设备event计时及B64实际编译产物。源码预检查通过，GPU未运行。当前唯一GPU仍.cv正式4K，之后已有max_num_seqs64服务排队，必须按实际进程结束和GPU空闲串行安排本实验，不能抢占。
