#!/usr/bin/env bash
# Sourced only by the qualified builder; scores and base are its validated inputs.
{
 cat <<'EOF'
# 天数 BI-V150：QKV/O 预填充原生INT8，r3

原始 MiniCPM5-2B checkpoint与结构不变，没有训练或微调。在已发布r2基础上，QKV仅M128/512/4096、O仅M512/4096使用原生A8W8；M64仍是BF16，保留原Down/GU/LM/KV和框架优化。r2的890项模型输入保持原字节，新版本903项模型输入独立冻结，旧标签不覆盖。

EOF
 printf '正式4K（4096/1024/C64/N256，4轮排除第1轮）：**%.2f total tok/s、%.2f output tok/s，TTFT %.2f ms**。后三轮极差/均值%.4f%%。相对上一版正式5911.45提高%.4f%%，TTFT相对3305.80ms变化%.4f%%；历史正式比较存在时段差异。\n\n' "${scores[0]}" "${scores[1]}" "${scores[2]}" "${scores[3]}" "${scores[4]}" "${scores[5]}"
 printf '同请求 N64 短对照净增%.4f%%，候选热轮波动0.8591%%；第三组相同框架BF16控制验证量化增量约1.2978%%，不能把全部收益归于单个kernel。短测不是正式N256成绩。\n\n' "${scores[6]}"
 cat <<'EOF'
新鲜完整Math-500 Level3：C8 **101/105（96.19%）**，C64 **102/105（97.14%）**，包括达到输出上限的所有请求，逐条原始评审独立重算，均严格>95%。C64已实际执行QKV/O各42层M4096原生路径，84个M64捕获投影均BF16；Python路由计数不冒充图重放次数。原三项BF16 GEMM预热、全部继承运行检查、新Down/QKV/O审计、1024请求成功、零抢占和服务就绪后无Triton autotune均通过独立CPU核验。

新增397135872 bytes（约378.74MiB）单一NN布局QKV/O打包副本，保留原BF16；和已有Down约504.33MiB均计入真实DeviceMemoryProfiler及峰值，未手填KV预算。正式实际KV604784tokens，原r2正式621440。没有新16K结果，不能当作完整双项比赛或整个资格已审核。

在天数容器中执行：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_generated_sources.sh
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_controller_evidence.sh --check
TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN=1 bash experiments/tianshu_native_int8_attention_prefill_release_20261022/serve_tianshu_minicpm.sh
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/serve_tianshu_minicpm.sh
```

六份必要控制JSON缺失时可无参数运行restore_controller_evidence.sh，仅从随版gzip恢复精确字节，已有文件SHA不同则拒绝覆盖。模型、依赖、编译缓存和原始数据在容器，GitHub仅代码、报告、关键JSON/CSV及必要压缩控制证据，未上传题目、预测全文或数组。启动前逐项验证模型和控制器SHA，只重定向记录输出路径，清理继承实验环境，原始13serve参数完全保留。

EOF
 printf '实际启动校验%s项模型输入、%s项控制器、63项环境。质量见相邻tianshu_native_int8_attention_quality_results_20261022；正式成绩见tianshu_native_int8_attention_official_results_20261022；三组短测见tianshu_attention_three_arm_results_20261022。\n' "${scores[7]}" "${scores[8]}"
} > "$base/README_32g.md"
