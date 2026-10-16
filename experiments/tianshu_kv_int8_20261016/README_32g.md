# 天数INT8逐token/head KV方向核验

已只读核对实际安装源码，原生支持int8_per_token_head；只是确认存在，尚未运行GPU或获得收益。Query/权重保持BF16，KV存储改int8并按token/head保留FP32 scale，会有量化误差，不能复用当前102/105质量结论。

直接切dtype会丢失现有hybrid/nativePrefill和.cv Decode准入，不能把回退后的速度当作公平对照。原生inline132的对齐与scale抽取也需要检查。先独立组件ROI，对原BF16与dequantized整数参考分别衡量；通过后再做真实writer/布局兼容、服务对照和完整Math>95。只有明确提升才做正式4K，无16K。

详细源码SHA、已核对事实和下一实验边界见FEASIBILITY_32g.json。当前正式最佳4093.81/Math102不变，GPU硬停止23:58。

第一轮attention组件已完成并复核：B64/4K packed128慢7.06%、inline132慢20.18%，不采用，尚无真实writer/服务/Math。独立载入反量化变体另见probe_attention_dequant.py，准备源码98a6d26。

载入反量化变体也已完成并本机复核64源码/105事件：B64/4K对齐w1慢26.46%、w2慢21.32%、inline132慢33.31%，不采用。两个候选均不进入真实writer/服务/Math。
