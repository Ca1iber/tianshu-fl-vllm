# 原循环候选首次启动：参数绑定失败

首个融合调用把新增noise形参位置绑定错，原生heuristics缺少N，抛KeyError(N)。未启动融合kernel、未完成case或性能轮次。原源码和结果不改，native_v2只修复为显式具名参数，设备生成源码SHA完全相同，并增加CPU AST绑定检查后完整重跑。
