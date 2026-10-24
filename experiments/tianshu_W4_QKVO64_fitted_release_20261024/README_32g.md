# 天数BI-V150 r5：拟合W4 QKV/O64

正式4K（4096/1024/C64/N256）：**6123.31 total tok/s / 1224.66 output tok/s，TTFT3202.70ms**。4轮剔除首轮，有效三轮6124.78、6138.44、6106.71，极差0.5182%，1024请求全部成功。相对r4历史正式均值吞吐+1.1960%、TTFT约+4.10%；同场新输入N128配对吞吐+1.0429%、TTFT-1.7755%，两种比较分别记录。相对官方2028.01吞吐基线约3.019倍。

全新Math Level3：C64 101/105、C8 102/105，均>95%；105全部成功且max_tokens计入分母。原始checkpoint完整SHA、模型结构均未改，没有训练或微调。此版为4K吞吐版本，没有新16K成绩，旧r4保留。

r4已有Down64/LM1..64原生W4及GU W8A8融合等设置保留；本版增加84个QKV/O的G128拟合仿射W4 buffer，只在符合范围的captured uniform64 FULL decode Graph中使用。其他M/prefill回退父版。原BF16/W8副本保留，额外200.8125MiB载荷计入真实profile，不宣称净显存节省。完整997模型来源、65环境、真实Bash导出、原始13参数命令均已核对。

在已准备好的天数容器复现，模型、SDK、权重副本、固定384算子缓存及原始评测留容器；GitHub只保存代码、报告、关键结果和SHA。先检查必要来源，再直接运行：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_generated_sources.sh
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
sha256sum --check experiments/tianshu_W4_QKVO64_fitted_release_20261024/MODEL_SOURCES.sha256
sha256sum --check experiments/tianshu_W4_QKVO64_fitted_release_20261024/W4_PARENT_PAYLOADS.sha256
sha256sum --check experiments/tianshu_W4_QKVO64_fitted_release_20261024/CHECKPOINT.sha256
source experiments/tianshu_W4_QKVO64_fitted_release_20261024/environment.sh
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

source可传入一个新的绝对输出目录。脚本只重定向正式65项环境中的记录路径，不加载Python/GPU；启动时从原权重确定性重建84个W4 buffer并逐bank核对既有SHA。FITTED_ATTENTION_BANKS_32g.json保存原权重/打包/scale/zero的元数据，没有复制大数组。父版45个W4载荷仍由W4_PARENT_PAYLOADS.sha256逐文件核对。

若SHM重建后要重新运行实验控制器，三个独立SDK证据引用按[SDK恢复说明](../tianshu_W4_QKVO64_fitted_results_20261024/SDK_RESTORE_32g.md)恢复，持久tar.xz仍在容器；不改安装库。正式报告与所有四轮关键结果见[完整4K报告](../tianshu_W4_QKVO64_fitted_official4k_20261024/REPORT_32g.md)，配对证据见[新输入筛选报告](../tianshu_W4_QKVO64_fitted_fresh128_20261024/REPORT_SCREEN_32g.md)。
