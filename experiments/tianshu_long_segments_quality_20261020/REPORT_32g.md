# 长上下文S16方案：全新Math Level 3为102/105（97.142857%）

2026-10-20，天数BI-V15032G，原始serve argv，K8V16、4096预算、83预热点、原有严格TopP32K和M2048配置；长上下文仅B33..56使用S16。新的105题Math全部执行并独立核验逐题review/prediction：**102/105=97.142857%，严格超过95%**。没有沿用原配置的旧Math；105个唯一index、105请求成功、0错误、全部105条自然stop，没有max_tokens截断，错误题index18/83/87。

评测使用用户原始evalscope命令，Level 3、batch8、temperature1/top_p.95/max_tokens32768、timeout3600。服务实际46项环境、225份源码/快照、新内核及真实FL Graph捕获/重放审计通过。Math规定batch8；高并发S16路径的GPU数值/保护区验证和服务同图重放由前置组件及N64对照提供，不声称本Math覆盖所有高并发输入。

同一原始serve argv、相同225份冻结源及归一化后相同46项环境已进入4K/1K C64 N256与16K/1K C64 N128完整正式测试，各四轮排首轮，尚未取得新正式均值。当前正式最佳仍4K5101.45、16K3213.11。前置小规模16K对照均值3386.67→3751.55（+10.774%）不能替代正式成绩。

GitHub只保存原始EvalScope汇总报告、完整质量审计、105题index/评分/token/stop及预测行哈希；原始答案、review全文、服务日志和编译缓存均留容器`/workspace/benchmark-results/20261020_long_segments_math_tianshu`，原文件SHA/大小在QUALITY_ARCHIVE_32g.json中。没有修改模型权重、官方benchmark或沐曦仓库。
