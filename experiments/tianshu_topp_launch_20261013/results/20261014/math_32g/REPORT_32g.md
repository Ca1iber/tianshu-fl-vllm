# 天数 BI-V150 32g：新的TopP启动配置独立Math-500通过

新TopP w16/stage1配置从头运行官方Math-500 Level 3全部105项，103/105（98.095238%），105请求全部成功、0错误、不完整false；错误题号18/87。所有105项自然stop，没有max_tokens截断。准确率严格超过95%，本次结果属于新TopP配置，不复用旧Decode L0/KV32的103/105。

计算基础与当前正式最好3426.62的Decode L0/KV tile32/B75G1/budget2048一致，只改变native TopP launch参数；函数体、grid、blocks、温度/top-p/top-k和RNG保持原函数。六份已安装计算源和coherent hook均匹配新的五轮服务候选。实际TopP路由1项，w16/stage1，source SHA64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8。Decode有13组producer/reducer同tile32配对证据，generated source4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e。

外层run_math_eval.py只为API子进程注入真实TopP hook并记录源SHA/路由，内层已验证的coherent Math runner重新执行官方EvalScope命令。foundation service root用于验证未变化的Decode/Prefill源；matched_topp_service_root独立指向本次confirm5_retry候选。报告没有将foundation旧服务6.82%当作本次TopP收益。本次TopP五轮同期小测为3458.33→3533.78，+2.181653%，640/640成功；正式4K在本次质量通过后新启动，仍待四轮完成。

逐项核验105唯一prediction/review索引、成功状态、错误字段、正确数、每份原始文件SHA、全部分片SHA和整包SHA。原始包3197095bytes、13分片，SHA256 b1b7e4fb48794586d590dac564da2fdd56e09ca93d70300c86d7a98c916a2af7。源码快照与本检查点run_math_eval.py及两个hook逐字匹配；原件service-results/20261013_tianshu/topp_launch_math/math。准确率与旧质量同为103/105，不据此声称准确率提高。

质量结论：通过，允许进入未修改的官方4K输入/1K输出、C64/N256四轮排首轮验证；不追加16K。正式最好暂仍3426.62/TTFT4514.45ms，较官方基线2028.01提高68.96%；未将小测3533.78升级成正式成绩。性能剖析：TODO，尚未采集。安装源、权重、官方benchmark不改；沐曦只读。
