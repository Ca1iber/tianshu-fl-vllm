# 天数 BI-V150 32g：原始serve命令验证及N64缓存效应

服务启动命令逐项与用户提供的官方命令一致，不额外传预算/maxseq/stream/threshold等参数，也未在代码里改写这些scheduler配置。真实Worker预算2048、maxseq256、async=True、modelmax131072；实际75点预热到2048。保留现有CV/KV16/w1/S8、native Prefill L0、预算内padding/int32 gather、原生TopP及原采样。

小测使用未改原benchmark脚本的4K/1K/C64/N64场景，四轮预排首轮。256条请求均成功（原stdout/CSV，脚本不产出逐请求正文）；44份源码与快照、SDK默认参数源码、实际Worker/75warm/算子路由及全部metrics零preemption均本地核验。

|轮次|total tok/s|output tok/s|TTFT ms|成功请求|
|---|---:|---:|---:|---:|
|1（排除）|3643.77|728.75|19592.13|64.0|
|2|5522.14|1104.43|1089.67|64.0|
|3|5516.12|1103.22|1080.53|64.0|
|4|5491.98|1098.4|1060.92|64.0|

原脚本给出后三轮5510.08 total tok/s、1102.02 output tok/s、TTFT1077.04ms。这是重复输入被前缀缓存命中的N64结果，**不是正式N256成绩，也不能作为可直接迁移的吞吐提升**。本轮只据此确认服务原命令和算子组合可运行，不采用为比赛新最佳。

四轮client日志均seed=0、temperature/top_p为None（使用服务默认）；metrics before为0，after查询1048576tokens、命中783360tokens，累计命中74.707031%。N64输入262144tokens能落入实际KV容量522912，而N256输入1048576tokens超过该容量，缓存行为不能视作相同。运行中额外观察metrics也保存在原始包observations中。后续小规模筛选回到每轮不同输入种子的实验，正式N256保持原脚本不改。

归档3855276bytes/15parts，SHA `d6d3e06edf72a7a36a035b2beccc844dbc3c58bff896a9adbda5a5298f3c0f86`，本机raw：`service-results/20261016_tianshu/original_command_screen/20261016_original_command_screen_tianshu`。本地复核：`python3 experiments/tianshu_original_command_20261016/verify_downloaded.py --root <raw-root>`。

此原命令配置对应的新Math105已在独立目录启动，不复用旧Math；严格>95%并本地raw复核后，再运行N256四轮。原自选参数4091.29保留为历史实测，不替代原命令验证。未运行新16K场景，不据单一4K承诺整套比赛资格。
