# 4096预算、83点预热质量链路预检

通过真实Math入口运行到拦截的首个vllm serve Popen，验证4096预算、threshold0、S8候选、SQL单表反射、83点扩展、buckets/gather开启及10份冻结源码。未创建服务进程、未查询GPU、未发请求。预检用明确标记的合成五轮fixture，仅用于覆盖分支，不代表吞吐或正确率；不能被GPU续跑入口消费。

新增串行续跑：等待独立5+5确认任务实际退出并完成归档，核验全部640请求，只有收益≥2%且候选所有计分轮均高于对照才启动全新官方Math105。严格>95%后运行原官方4K脚本四轮N256，排除首轮，基准最佳3826.24。所有任务受预约15:58 UTC截止约束。当前没有新的Math或正式4K成绩。

原官方4K入口也已通过真实main的CPU拦截预检：使用明确标记的合成confirm/quality fixture覆盖质量门槛、10源匹配、当前安装源码、预算与钩子组合，在首次serve Popen前终止；没有GPU进程、请求或性能结果。原基准脚本SHA不变。对应CPU_OFFICIAL_CLI_AUDIT_32g.json。

另外准备独立候选启动脚本serve_tianshu_minicpm_candidate.sh，真实bash exec被CPU桩拦截，CLI与预检Math入口逐参数一致，4096/83点、S8和共享SQL开关一致。仅候选工件，当前正式最佳启动脚本与tag未改。GPU实际Math/正式argv仍须完成后独立匹配。
