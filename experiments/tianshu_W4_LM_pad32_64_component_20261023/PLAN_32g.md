# 原生W4 LM M1..64补齐到32/64组件

复用完整已验证W4 LM bank及前版M32/M64完整参考输出；不换模型、scale、zero、quant格式。BF16 M4/8的旧失败明确保留，不把不合格控制作收益宣称。本组件仅比较相同W4精度的直接原生与补齐路径，衡量新padding成本，不重复宣传BF16提速或当模型收益。

任意M1..32补到32，M33..64补到64：单个Triton kernel只复制真实输入，其余填BF16+0，原生dotsaddz计算后返回前M行；32/64直接原生。没有全局Tensor workspace，非32/64调用含实际pad及native输出分配。CPU来源锁定：M32使用小批次v2实际全量输出；M64使用初版已验证LM64输出并roll1，输入/FP64参考同roll，避免M1零输入。

全部64个M×3phase：pad输出真实前缀逐字等输入、尾全+0、保护区不变；实际候选eager与NaN-poison Graph完整输出逐字等相应独立已验证bucket输出前M行，不只抽样。匹配时保留全输出SHA，CPU可从冻结bucket原数组独立重算对应前缀SHA；若不同先保存完整失败数组再中止，不覆盖旧实验。完整输入/bank/scales/zero版本和字节、guards、RNG/checkpoint/source不变。

独立CPU用原quantized FP64、bucket完整BF16输出，对每个M完整输出验证原逐格门槛与relativeL2；按行累积平方误差/参考能量、逐行最大(absdiff-.012absref)的前缀最大值，可等价核验全部前缀的absdiff<=.012absref+.006RMS与L2<=.006，不抽样、不放宽。引用bucket已验证完整数值不是模型Math质量。

8个预声明M各eager/graph7轮排首轮、后6中位数、20次调用，交替直接W4/补齐W4顺序，保存所有开销。LM实际图外，以eager为主；后续仍需模型C8/C64全105题>95%和独立短4K。GPU前冻结，900秒、15:58UTC截止；不做模型/正式benchmark/16K。大数组仅容器，关键源码/报告/结果入天数GitHub。
