# 仅 CPU 复核修复，不重复 GPU

原 GPU 记录器的原阶段文件被第二阶段覆盖，初次离线校验失败日志原样保留。新校验器检查冻结源码和两文件实际内容，并利用源码中明确的、精确自逆的词表维flip和动态P batch维flip恢复原阶段输入，重新读取所有完整GPU输出进行独立FP64语义验证。

不修改原GPU代码/CPU_GATE/原件，不重新执行GPU，不把恢复过程称为原记录器成功。主报告与关键结果在 `../tianshu_topp_online_sum_results_20261021`。独立运行：

```bash
env -u PYTHONPATH python3 experiments/tianshu_topp_online_sum_cpu_recovery_20261021/verify_gpu.py \
 --root /workspace/benchmark-results/20261021_topp_online_sum_GPU_tianshu \
 --output /workspace/benchmark-results/20261021_topp_online_sum_CPU_recovery_recheck_tianshu
```
