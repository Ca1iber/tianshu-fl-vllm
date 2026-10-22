# Down同时量化解码/预填充：新C64 95/105，拒绝

全部105接口成功，官方evalscope新得分95/105=90.476%，低于>95%要求（至少100/105）。错误索引77,87,18,23,83,72,32,26,7,101，32/26/7/101达到32768上限，全部保留分母。没有新C8、短4K或官方4K，未采用，r1正式5777.22 tok/s不变。

原实验同时留下audit_failed，不改成complete：旧base_runtime_audit第100行要求BF16 Down仍出现在原vendor-mm预热记录中；新Down opaque op的BF16 fallback调用eager F.linear，实际只留下O/GateUp两个vendor-mm记录。完整旧运行审核没有通过，不伪造缺失Down记录。CPU仅重新核对官方105评分/执行元数据、原890模型源/27控制源及其snapshot、真实新Down packing/profile/binary/42层runtime/capture证明。

后续修复在新版本目录进行：BF16回退直接使用torch.mm以恢复既有vendor-mm路径，保留原三项GEMM检查；模型首先只在prefill量化Down，M64 decode回退BF16。实际新路径须在模型启动后验证，重新全105 C64和C8，不能复用本轮结果。

原raw/日志留在容器 /workspace/benchmark-results/20261022_native_int8_down_all_math64_tianshu；原服务/控制器源码不改。这里只保存负结果关键数字及CPU核对，不声明原实验或完整运行审核通过。
