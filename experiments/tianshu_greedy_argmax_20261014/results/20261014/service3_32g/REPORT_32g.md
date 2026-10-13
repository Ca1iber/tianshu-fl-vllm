# 天数 BI-V150 32g：精确贪心 Argmax 服务无收益

4K输入/1K输出、C64/N64各3轮，预排首轮。对照3846.751238→候选3847.885980 total tok/s，变化+0.029499%，不采用，不追加Math或正式benchmark。两组都保持已验证最好版本的L0/KV32/warp2-B8..64、其余warp4、B75G1和budget2048；仅候选贪心Argmax改为两阶段归约，公开输出仍int64。此短测不是正式N256成绩；当前正式最好3640.16保持。

|轮次|对照tok/s|候选tok/s|对照TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1376.025681|1374.605374|13334.967034|13534.532805|否|
|2|3845.268092|3849.288341|13103.882452|13034.017712|是|
|3|3848.234383|3846.483618|13036.515999|13086.775142|是|

384/384请求成功，原始文本、逐轮seed、实际usage4096/1024/5120本地核验。两组temperature0/top_p1/repetition_penalty1/logprobsNone，与实际原始benchmark后端的贪心采样一致；旧随机短测参数未追改。Math仍应使用官方temperature1/top_p.95并对新源码独立测试，本候选没有新Math。7条贪心canary文本相同不替代Math质量。

请求阶段真实B64路由已验证：对照调用原Sampler.greedy_sample，候选命中23ace17a实现、tile2048/warp4，输出int64；B1使用tile1024。双组实际75点预热完成，七份冻结源、六份安装源清单、server argv一致；TopP记录为空符合真实贪心提前返回。实际模型的aten::argmax CUDA注册来自FlagGems，旧独立micro没有记录其Torch基线提供者，因此micro的88.7%局部改善不能直接解释成模型热路径改善。

原对照三轮已完成192请求，但包装器错误要求TopP记录非空，发生推理结束后的AssertionError。原failed状态及原始包保留。本次只去掉该后验断言的非空条件，其余GPU与请求代码逐字不变；新目录复用192份对照原件并单独补跑候选。原result SHA ce435ed877776a743f1263f0a589f6bd7f5a69f76dffed55c228e2a3ff6f222d，旧trial 3ea5aff0→新trial 73f67ba2。control_recovery_audit.json及本地audit_32g.json证明原件逐字一致、旧失败状态保留、唯一源码差异在推理后的验证。新control/result仅增加75点及重验字段，没有改原始分数。

全原始包3203724bytes/13分片，SHA256 4f8ba6b507f75f1ddaacd4261bb62a57240a9f0404f1f465139975d7f16bf283。分片/整包、384响应、七份源及复用证明均已本地核验。原件service-results/20261014_tianshu/greedy_argmax_service_recovered/20261014_greedy_argmax_service_recovered_tianshu。失败原包另存postvalidation_failure报告，不伪装成完整成功实验。

结论：算子正确，但本工作负载无服务收益。硬件profile：TODO有效图内kernel覆盖；现有ixkn失败诊断不能给出内核占比。未改安装源、权重、原benchmark或沐曦仓库；未跑新16K。后续先核对算子提供者，继续寻找vLLM与算子机会。
