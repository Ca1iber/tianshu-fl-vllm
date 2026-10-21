# CPU恢复只校验已保存阶段证据，原实验仍失败

执行 `env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' python3 experiments/tianshu_stage_prefill_int8_failure_recovery_20261021/recheck_partial.py`。不运行GPU，不改原目录，不生成缺失metrics，不调用缺少必要输入的完整runtime审核。

`complete=true`只表示本恢复校验完成；同时强制`original_experiment_passed=false`、`canary_equal=false`、`full_runtime_audit_completed=false`。40帧区间与128份usage有效性、原407源子集与实际62环境、快照、输入seed/usage以及前后文本差异均重读验证。阶段数据只作诊断线索，未形成新质量或性能成绩。详见原测量目录REPORT_32g.md。原结果和源码冻结保持，完整raw仍在容器。
