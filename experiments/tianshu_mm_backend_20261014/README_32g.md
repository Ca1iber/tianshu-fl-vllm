# 天数32g：矩阵乘API诊断准备

对比显式FlagGems.mm与torch.mm：M8/16/32/64的QKV、O、gateup、down及词表130560的LM head，另补M2048四种隐藏投影，总24例。目标是确定现有矩阵乘路径是否还有明显空间；没有将任何API替换接入比赛模型，也没有据此宣称可采用的后端或服务收益。

两组BF16输入和NT权重相同，均使用API分配输出，不给一组额外使用out参数。禁用TF32与BF16 reduced-precision reduction，要求FlagGems没有启用全局ATen registrar，保存ATen mm dispatch表与源码/版本。完整GPU FP32和8行×32列CPU参考、Graph捕获与改变输入后重放检查，7轮AB/BA设备事件计时。编译、数值或Graph失败逐例保留，不把失败配置计作有效结果。

当前仅通过AST检查，GPU结果待定。控制器等待Decode补齐服务结束、归档及GPU/端口空闲后运行一次诊断；prefix或padding任一小测达到2%则先等待其确认，不抢占质量/正式评测。没有16K、安装文件修改、模型路由或新正确率结论。
