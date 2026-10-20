# CPU结果选择路径修正

第一条验证队列在 screen_gate 阶段因复制的 raw-root 仍指向上一轮348源目录而自动拒绝（本轮要求359源）。没有启动Math或正式GPU评测，也没有把旧结果当作本轮结果。

现已将结果根指向本轮LM precision control/candidate，实际运行完整screen复核通过。原失败runner/log保留，后续使用runner_v2，不覆盖失败状态。所有本轮GPU服务原始结果和359运行时源不变；控制器在新的Math启动之前更新并重新冻结。
