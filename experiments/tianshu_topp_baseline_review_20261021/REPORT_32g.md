# 当前 TopP 基线复核：重复方向，服务启动失败，未取得新增成绩

2026-10-21 10:30 北京，431-source TopP 服务控制臂初始化失败。原件 `/workspace/benchmark-results/20261021_topp_warp_service_control_tianshu/{server.log,result.json}`；错误为 `AttributeError: 'Router' object has no attribute 'fn'`，位于新 policy.py:27。服务没有进入 benchmark，零轮成绩，候选臂未运行，GPU 已释放。保留原 failed 判定，原实验源、计划、CPU_GATE 不修改。

新策略错误地假设 native._topk_topp_kernel 仍是未经包装的 Triton JIT。实际推荐版 407-source/62-env 已通过 `tianshu_gemm_topp_geo_20261016/hooks/top_p_sitecustomize.py` 包装为 Router。其候选标志 TIANSHU_TOPP_PRECISION_SERVICE_MODE=candidate、TIANSHU_TOPP_LAUNCH_PROBE=1 已开启。支持的 FP32、130560 词表、B8..64、词表维连续调用原本已经使用 num_warps=16、num_stages=1；纯 TopP 进一步使用实际最大值归一化、32 步几何区间搜索和 BLOCK_SIZE=32768/BLOCK_SIZE_TRUNC=16384。生成源码 SHA256 为 09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d。

10 月 21 日两布局原生 TopP 组件相对 installed 原始 w4、8K/4K 内核的 28–35% 耗时下降是有效的局部测量，但这个对照不是当前推荐版已使用的内核。因此没有新增的推荐版加速证据，不能将其换算为模型吞吐。逐位不等、CPU 语义差异和全部原始失败判定继续保留。不会通过解包 Router 并绕过当前几何搜索来假称原算法不变。

`STATE_20261021_TOPP_COMPONENTS_DONE.md`、`NEXT_TOPP_SERVICE_PLAN_20261021.md` 和失败实验 PLAN 中将 native w4 当作当前运行基线的下一步建议失效；这些是历史记录，不再执行。后续实验必须以实际活动的生成内核和真实服务配置为对照。

旧预筛选 0.5σ²/1σ² 已完成 GPU，普通分布无收益、尖峰明显回退，不能根据其旧 README 的“准备完毕”再次运行。6144 和 4352 旧预算也已有小 4K 结果；新精度配置下的复测须说明新假设，不能称首次探索。

当前推荐仍为 tag `tianshu-original-command-4k-prefill-int8-20261021`：正式 4K 5586.28 total tok/s、1117.25 output tok/s、TTFT 3780.63 ms、Math 102/105。新 8192 正式 5608.40 但波动/TTFT 更差，尚未采用；旧 16K 3885.91 不能算当前配置的 16K。
