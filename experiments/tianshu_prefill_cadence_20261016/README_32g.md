# 天数有界prefill调度节奏

保持原serve参数、2048预算、256最大序列与当前最佳算子。对有prefill压力的简单FCFS/async/MRV1/TP1 MiniCPM decoder，用原Scheduler已有next_decode_eligible_step检查最多连续推迟两步纯单token decode，每第三步走原策略；无prefill、低KV余量、显式throttle_prefills或其它不支持场景全部原策略。只临时改变本次调用的eligibility，在finally恢复；不更改请求队列、token数、KV分配、取消/抢占逻辑或输出。

目标是减少小decode与prefill交错造成的组批成本。可能增加流式输出间隔，必须同时记录吞吐、TTFT和请求wall/后续TPOT，不能只看吞吐。先CPU合同/边界/例外恢复，再原命令新鲜4K/1K C64/N64 3+3服务，预排首轮；有明确收益再新Math105>95与正式测试。没有采用，不改最佳4093.81/Math102。

无16K、安装源码/权重/原benchmark/最佳启动器与沐曦改动；2026-10-16 23:58 GPU硬停。

V1已中止：配置flag为True，旧guard无命中；只完成control首轮，不是性能A/B。原始失败包与69源/64响应已本地审计，详见results/20261016/guard_mismatch_32g；V2/V3为独立目录，不改运行源码。
