# 天数有界prefill调度V2与MRV1 token恢复

保留V1未命中守卫的冻结源码和中止记录，不重写为通过。真实EngineArgs预检显示use_v2_model_runner=True；但WorkerFL显式构造ModelRunnerFL，其跳过一轮后重新加入请求依赖cached_reqs.all_token_ids。原Scheduler仅flag=False时填充该字段并维护prev_step_scheduled_req_ids，因此不能简单放宽V1守卫就暂停decode。

本版严格守卫WorkerFL、TP/PP/DP/CP1、无Spec/LoRA/MM/Mamba和MiniCPM geometry。候选仅在原schedule调用期间选择已有MRV1元数据路径，finally恢复配置flag；同时每三步最多让出两步decode给prefill，保留原队列、预算2048、分配及取消/抢占逻辑。控制沿原flag与原调度。这是一个包含必要元数据兼容的策略包，不能把结果归为仅节奏改变。

CPU检查含真实Scheduler签名/eligibility/cache recovery源分支、WorkerFL构造调用、12步节奏/异常与既有eligibility恢复、两种配置flag下元数据选择与恢复。仅stub合同验证，不是完整调度模拟。服务首先确认实际guard命中，并用两轮C8/4K/16 greedy并发探针，候选必须实际推迟并传递恢复history，32探针跨组文本一致才可比较；探针不替代Math。

随后同原serve命令与最佳CV/KV16/S8/GEO/M2048配置，新鲜4K/1K C64/N64 3+3，预排首轮，记录全部响应/TTFT/客户端decode均摊间隔和文本事件间隔。策略最多连续两步，不保证绝对毫秒延迟上限。仅明确收益才新Math105>95与正式benchmark。没有采用，最好4093.81/Math102 unchanged。

无16K、安装源码/权重/原benchmark/最佳启动器或沐曦改动，23:58硬停；中止时强制关闭自有API组避免HTTP请求与graceful shutdown相互等待，正常结束仍先TERM后KILL。
