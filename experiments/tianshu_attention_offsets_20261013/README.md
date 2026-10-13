# 天数32g Decode地址算术32

冻结native attention源仅整数element offsets/strides改32，pointer types和float body不变。实际共享交错KV和Q stride的10场景与原64逐位一致，FP32参考通过；中等batch收益4%-24%，B1/B64无收益。严格检查带masked tile余量的元素offset<2^31，服务仅B4..48满足geometry时选择32。

报告与两份完整rawSHA：results/20261013/micro_real_32g/REPORT_32g.md。helper/probe已GPU微测验证；servicehook/harness及16K双组服务实验仍在跑，未采纳，不把micro当正式吞吐。所有安装源/benchmark未改，沐曦只读。
