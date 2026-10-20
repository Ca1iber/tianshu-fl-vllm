# 预算V1候选在启动阶段被旧保护拒绝

V1只有control完成三轮小4K，candidate8192没有进入benchmark，不能给出8192性能或完整pair通过结论。新Math条件队列看到pair失败后退出，未启动Math或正式N256。

control4096：5227.530768（首轮排除）/5419.374806/5436.665016 total tok/s，TTFT12915.521633/10888.155143/10883.788254ms；后两轮均值5428.019911，六位近似spread0.318536%，每轮64成功，三轮autotune均0。完成后的真实runtime audit已重新独立执行，并与保存的profile_audit完全一致。其余源码和完整响应留容器。

候选实际EngineArgs已选择8192、seq64，原serve argv没有改。EngineCore在2026-10-21 02:17:05被 `tianshu_long_segments_service_20261020/policy.py:40` 的 `assert max_num_batched_tokens==4096` 拒绝，尚未加载模型进行吞吐测试。不能将此次启动错误报告为8192负优化、OOM或模型正确率失败。

V2在独立目录复制分段policy，仅把该预算期望表达式改为严格的mode→4096/8192映射。归一化此表达式后整模块AST与原版完全一致。新hooks仅临时将精确的旧long hook路径路由至独立副本，注册完成即恢复runpy；真实worker源/哈希/预算/seq会被记录和审计。原合格文件未更改，不用临时伪造4096绕过检查。

V2两臂都从新的根目录重跑，不借用V1 control作为其完整pair。V1原始目录为`/workspace/benchmark-results/20261021_dense_budget8192_service_{control,candidate}_tianshu`，失败日志、源码快照和条件队列失败状态均保留。PARTIAL_GATE只声明control审计和candidate启动失败，不声明完整通过。
