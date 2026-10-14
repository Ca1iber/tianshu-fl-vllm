# 天数BI-V150 32g：Decode .cv全新Math500 Level3 105题

对应.cv独立5+5确认的同一真实进程配置，未复用旧正确率。105请求均成功、0模型调用错误、0不完整，正确103/105=98.095238%，严格大于95%。错误题18、87。

104题stop，1题index35达到官方generation-config设定的max_tokens32768，评分为正确。没有手动中断最后一题；不能把本轮写成105题全部自然stop。temperature1.0、top_p0.95、max_tokens32768、eval_batch_size8、官方本地Level3数据路径与服务配置已逐项复核。

总耗时含启动/清理1561.07秒，开始2026-10-14T11:52:33.675380+00:00，结束2026-10-14T12:18:34.741221+00:00。不同随机生成长度影响总时长，本报告不将其当作与旧Math的吞吐对照。

本机已完整下载并复核105条prediction/review及唯一索引、逐题错误/评分、指标重算、13实验源与6安装源、实际83点预热、S8/KV16/原生TopP/SQL及reducer配对。实际14组Decode路由的.cv生成源8deb和原L0源4a1f字节重新计算SHA，并从原L0完整重现新源，来源链nativebab2→L0→.cv通过。并发8的Math不宣称覆盖B64实际题目，B64结合此前真实C64服务与startup/capture记录验证。

原始归档4326643bytes/17parts，SHA256 `7b972d575e857213b940733aaf1d9437cd97c048cb9eac333c65eb479f268ee9`。本机raw `service-results/20261014_tianshu/decode_cache_cv_math/20261014_decode_cache_cv_math_tianshu`。

复核入口：verify_downloaded_math.py --root <mathraw> --confirm-root <confirmraw>；audit_generated_kernels.py --root <mathraw>。质量验证已通过，原官方4K N256四轮skip1已实际启动；正式完整数据及启动脚本核对通过前，当前best4000.99保持。
