# Down64 + LM64 W4完整质量独立CPU核验

独立读取全部105题reviews/predictions/report，包括cap/errors完整分母，核对全部原始文件SHA、934来源及95控制器/快照、checkpoint、原13serve参数及实际63环境。重算全部继承与新增profile和accuracy审计并逐字匹配；Down42W4真实捕获与C64主模型图重放链接、LM64真实W4调用>=10次且旧W8 LM64计数为空，所有新buffer真实load/native scratch/存储计入。C8保留BF16 LM8，不能冒充已测试LM64。全部CPU bank文件重新SHA验证。

只验证实际已结束结果，不启动模型、不导入GPU运行库、不改raw或阈值。可先--batches64输出C64_门禁但完整资格始终false；默认必须新鲜C8/C64各严格>95%才model_quality_qualified。任何失败保留，不原样重跑追分。当前尚无数学结果。
