# 真实服务前：复现全部23个正式range控制cubin

3点来源诊断已复现正式完整cubin：M128/V1、M512/M4096/V2。实际旧range模型会warm全部23M，不能只准备两个native白名单点。新独立任务对当前service_v2 selected_plan全部23M，两个AST完全相同设备来源各独立cache，共46次零输入GEMM编译/执行。逐M必须完整SHA等于正式gate才选择，保留所有不匹配项；不替换或修改旧cache。

只证明控制binary与来源，不是模型质量/性能，不读checkpoint。全0整数矩阵、scale1，RNG不变，输出0；确认GPU空闲后，单任务600秒，当前北京时间12点左右，早于今日23:58硬停，每日06:00–24:00授权。最终选择的cache/group实际文件需再CPU hash冻结并用于独立模型接入。代码报告关键JSON推天数GitHub，设备cache只留容器，无本地副本。
