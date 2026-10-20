# 整数GEMM整模型接入诊断记录

- v1：导入FLModelRunner失败，实际类名是ModelRunnerFL。没有模型请求。原始目录20261020_dense_w8a8_service4k_tianshu。
- v2：43组权重加载且LM head实际执行INT8，但GateUp没有新调用。原生日志显示旧AOT图被直接加载；192个小样本请求成功，计分轮5222.80/5211.60，不作为本组合成绩。审计报(64,0)并阻止控制组/新Math继续。原始目录20261020_dense_w8a8_service4k_v2_tianshu。
- v3：两组统一禁用旧编译缓存，C8/C64均真实捕获42个GateUp，LM头执行INT8；layer0/LM原始权重、INT8和scale哈希与组件测试完全匹配。三轮小样本4171.25/5416.13/5397.97，192个请求成功、零现场autotune。但基线审计在base_runtime_audit.py:100发现M2048投影集合不完整；新opaque-op的GateUp回退调用F.linear，进入linear_kernel并绕过已测的NT mm优化。该结果仍不作为新组合正式或对照成绩。原始目录20261020_dense_w8a8_service4k_v3_tianshu。

v4把不采用INT8的GateUp回退改为flag_gems.mm(x,weight.T)，对齐编译后原有BF16 mm调用及M2048已调tile。LM head其他批次保留原EmbeddingMethod.apply，仍排除输入embedding。权重未修改，INT8配置不变；在独立v4目录重新做完整小样本对照，必须保留全部原审计要求。

所有失败原始来源快照、日志、状态和完整回答保留容器；Git只保存代码与关键诊断，没有清理全局缓存，也没有放宽或伪造通过标记。新Math尚未启动，正式5235.52版本继续保留。
