# CPU核验计划：BF16大M扫描

不启动GPU，不重跑已完成GPU。1008唯一role/M/config/phase全覆盖；三个role真实第0层checkpoint及公开固定seed重建A，保存GPU实际输入选点逐位匹配，F64 CPU重算前后所有四路选点。每个候选完整输出在线数值检查和CPU选点均通过，实际compiled binary在真实Triton缓存逐一核对，不合格native/current参考明确记录和排除。7轮计时从原始值独立求中位数，双相位均值对比真实current（保留M2048 O/Down special），同tile比较才归因W重排，跨tile收益标作tile+调度/布局整体效果。单层组件不能作为模型成绩，后续需42bank/原4K/新质量。代码与小型关键结果入GitHub，完整GPU数据只留容器。
