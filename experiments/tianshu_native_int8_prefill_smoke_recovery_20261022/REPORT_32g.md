# 接入 smoke 的独立 CPU 核验

原始 service_v2 smoke 保持 audit_failed，没有修改或覆盖。失败点是 baseline_pair_audit.py 的 admission_checks>=194，继承自三轮N64，而此次明确只跑单轮N64，另有2次canary和8次smoke，共74次。

独立核验通过：74个真实响应和token数、重复seed canary相同；全42层M4096在真实请求中命中新原生INT8路径，并在原生显存profile内实际执行；192MiB临时INT32工作区未手工移出预算；原M64全42层capture、全部23旧尺寸完整binary、新binary和877冻结源保持不变；本smoke波次没有就绪后autotune。

本结论仅说明接入正常，不代表性能或Math通过。原完整吞吐审计没有声称通过；后续三轮A/B必须原样通过其完整检查。CPU_RECOVERY_GATE保存原结果hash和源码图，原日志与响应留容器，不上传大体积原始响应。
