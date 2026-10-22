# LM审计接线修正，仅CPU，原数值与原失败结果不改

925模型输入和82原质量控制器已经在GPU前冻结。本轮发现原适配指向未使用的lm_precision旧副本；真实Math64路径是nn_silu_fx_c64_quality/dense_quality_audit.py，正式路径是nn_silu_fx_screen/dense_audit.py。模型本身、采样、原始105题和全部源字节保持不变，不能原样重跑挑分。

各复制真实审计文件，只将LM64旧kernel调用计数替换成新LM route的实际64整数调用计数，保留原replays阈值（Math64=10，正式=100）及全部其他GU捕获/重放、原pack/scales/显存/参数/环境/RNG/KV/调度/采样检查。新的LM pack/CUBIN/临时显存/实际batch范围由原新服务audit_profile另外完整核验。C8仍走原LM8 BF16检查，不要求未命中的LM64在自然Math期间出现。不能伪造旧binding。

若原C64在旧LM guard处audit_failed，只CPU重现确切原失败文件与语句，原runtime source/controller/snapshot与report/review/prediction完整核对后，使用修正分支重新运行完整原r3+新LM运行时审计和完整105题计分。sys.setprofile证明新Math分支实际被执行、旧Math LM guard不再执行；保留原parent组合，包括NativeAdmissionAudit。恢复输出在独立目录，phase=CPU_recovery_complete、original_experiment_passed=false、原raw phase/hash明确记录，绝不把原失败伪装成complete。新恢复控制器源和快照另存，原82控制器也保留。分数<=95%仍是不合格，不因审计修正变成合格。

只有恢复runtime全部通过且本次C64真实105题>95%，才允许原冻结C8控制器第一次运行；该C8模型925与原82控制器完全相同，使用原C8 BF16审核，不能借用旧C8分数。后续短测/正式必须识别恢复状态及原失败来源哈希，最终独立质量核验重新核对全部原文件及恢复控制器。不执行新GPU/改模型/原源码，不绕过其他守卫，新增CPU版本在任何后续GPU前冻结。
