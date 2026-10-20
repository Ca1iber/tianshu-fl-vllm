# 小暂存区服务结果：4K持平，16K运行中

verify_service.py在服务结束后独立核验两组全部384请求和token usage、准确配对输入、seeded输出、所有冻结源码及快照、真实FL Graph、新S16绑定、实际512/128MiB arena字节及内存profile调用栈，并重算吞吐与波动。结果写为SERVICE_4K/16K_GATE_32g.json和逐轮CSV，完整响应留容器。当前没有收益结论、没有新Math或正式成绩。

4K已完成独立核验：5031.29974→5030.59406 total tok/s（-0.01403%），384/384成功、0抢占、无容量fallback，实际KV增加12480 tokens。详见[4K报告](REPORT_4K_32g.md)。16K尚无计分均值；前述“等待运行”为目录建立时的历史记录。
