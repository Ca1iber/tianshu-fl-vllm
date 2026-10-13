# 天数32g：Prefill stage2服务筛选

同一时段新服务A/B，保持Decode地址32/B75G1/Native L0 M256/KV128/w16/budget2048，只把Prefill launch stage1→2。16K输入/128输出、C32/N32，每组五轮，排除1/2比较末三轮：5241.451594→5524.074349 total tok/s，+5.392070%。这是Prefill-heavy本地筛选，不是官方1K输出成绩。

|轮次|stage1 tok/s|stage2 tok/s|计分|
|---:|---:|---:|---|
|1|3605.959170|3753.856194|否|
|2|5241.014265|5530.699624|否|
|3|5246.874847|5530.031937|是|
|4|5243.686500|5518.461479|是|
|5|5233.793435|5523.729633|是|

两组全部320请求成功，每条usage16384/128，配对种子一致，7greedy探针逐字相同。kernel body、网格、float dtype、64地址Prefill/32地址Decode、其余安装源/hooks都相同。新override严格校验oldNativehook983/Hyb5c SHA及dispatch/router共享配置引用，仅将num_stages置2；真实模型route记录stage2已核验，不是只设置未生效的环境变量。

扩展8场景16变体、96独立CPU/GPU FP32、保护尾部/Graph和与stage1逐位一致都先行通过。新stage2独立Math尚未运行，不能用旧102/105认证新launch；已经启动正常16K/1K C64/N64同期三轮排首对照，确认真实输出长度下的收益。若仍明确提升再进行新质量与原始benchmark，暂不改正式best。

完整raw3147506bytes/13分片、整包SHA256 b87219a432fc593daeeb831826fdde72ef5c58b62b8d5feda3486ffccc83997e全部本地verified。执行source/helper/harness、全部320响应和路由保留在service-results/20261013_tianshu/prefill_stage2_prefillheavy_service；localsource逐字对应snapshot，没有改安装源码或原始benchmark。
