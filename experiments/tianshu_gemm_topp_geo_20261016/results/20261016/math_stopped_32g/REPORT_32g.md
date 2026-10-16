# 天数几何TopP组合：Math按时截止，保留103题终态原件

2026-10-16 17:42:03开始全新Math500 Level3全105题验证；temperature1.0、top_p0.95、max_tokens32768、batch8不变，原始官方serve argv及默认2048预算/256序列/75暖点。使用与4K短测相同的几何TopP32K及M2048三项GEMM候选，不复用历史Math结果。

17:58预设SIGALRM生效，控制器phase=stopped、error=14。完整写盘103条预测及103条评分，无重复index或截断JSON行，缺index83、101；已完成条目的评分状态均success（表示评分正常完成，不表示回答全部正确）。尚未取得全105题质量结果，不报告本版完整正确率，不删除已有错题，不采用旧103/105替代。

17:58:04终态归档确认runner47083为Z、API47085和evaluator47755已消失，ixsmi无GPU进程。全部58份运行源码和快照SHA均未变。全体原始日志、预测、评分、生成内核和源快照作为真实terminal_incomplete保存，无phase篡改。

本机原件：`service-results/20261016_tianshu/gemm_topp_geo_math_terminal/20261016_gemm_topp_geo_math_tianshu`。压缩包1089022bytes、5分片，SHA256 `8f26750a4f687175e1460b1326c44e4b0dbfe4433b9e984f10a309bdbbdbc22c`。分片、全包和每个捕获文件SHA、58源快照及当前仓库源、103题prediction/review一一对应、进程终态和GPU空闲均已本机复核。

```bash
python3 experiments/tianshu_gemm_topp_geo_20261016/verify_incomplete_downloaded.py \
  --root ../service-results/20261016_tianshu/gemm_topp_geo_math_terminal/20261016_gemm_topp_geo_math_tianshu
```

后续有预约时段时，需要同一候选完成全部105题。已只读保存容器EvalScope缓存实现供离线核对，尚未实施续跑，不预先声称缓存可用。若续跑，必须保持全部103题记录（包括错题）、唯一题号、数据指纹、候选源码和采样参数一致，并重新核对105条有效记录；无法证明时重新跑全套。此候选的正式N256×4测试尚未启动。当前完整正式最佳仍3992.97 total tok/s、798.60 output tok/s、TTFT4443.93ms，质量属于原版本103/105。

用户要求的上一版GitHub推送已完成：vLLM分支2389dd33、ops分支4cdc3b76，最佳标签指向9695cb6。本次后续探索和本报告仅本机保存，不另行推送。
