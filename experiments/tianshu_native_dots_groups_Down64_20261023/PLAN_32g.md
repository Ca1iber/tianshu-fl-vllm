# 原生dotsaddz W4 Down64 group64/128/256

在安装库已保存的二进制契约明确接受32/64/128/256后，比较64/128/256，不盲测不支持的值。全部42层真实Down weight、相同M64/K6144/N2048、相同3个GU派生activation fixture、相同行roll。对照包含当前r3实际BF16和原生group128 W4；原group128 bank/参考逐SHA复用，新group64/256 CPU按BF16 scale->RTN/clamp->自然N pack，全部42层逆解码和zero逐值复核，不修改checkpoint。

每个新group三层0/20/41×3phase完整K FP64参考，反量化权重先BF16舍入；使用冻结阈值absdiff<=.012*abs(ref)+.006*RMS且relativeL2<=.006。全部42层×3phase graph清NaN重放/eager逐字节一致，输入/全部bank/scale/zero全字节版本guard、RNG/source/checkpoint不变。记录每group相对原BF16的量化投影误差，静态误差不等于模型质量。

每模式graph/eager7轮、排首轮、后6轮中位数，每轮10次全部42层真实bank；循环旋转且交替反转顺序。Graph为主要判据；eager含不同分配/调度差异，不当作模型收益。保留三种group全部结果；best-of-3标明。仅组件，不接入模型、不借用现有Math质量、不跑完整benchmark/16K。独立CPU复核完成后才判断是否值得模型集成。

新bank/数组只留容器；代码/报告/关键JSON/CSV推天数GitHub。首次GPU前冻结全部源、CPU门禁；GPU最多900秒、15:58UTC前停止。任何设备/数值错误保留失败，不修改原冻结版或原样重跑。
