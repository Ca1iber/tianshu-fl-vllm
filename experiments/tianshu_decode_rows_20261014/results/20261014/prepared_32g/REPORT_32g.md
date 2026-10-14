# 天数32g：准备较大Decode查询行tile诊断

BM8虽然17例通过，但长上下文比BM16慢49%至89%；此处检验BM32/BQ4保持原warp、BM32/BQ4/warp2和BM64/BQ8/warp2。保持KV页16、当前guardedKV16/B24回退32和B64S8；实际grid必须随BQ重算，reducer KV tile同步。

CPU实际Router与生产LaunchRouter执行覆盖B1..64、稠密及两种0/1查询长度、BM16/32/64，共573个非全空组合，逻辑query无遗漏或重复，reducer tile一致。GPU17场景×3候选待跑，必须检查独立FP32/CPU、Graph改变query后的输出及尾部、真实编译元数据，保留失败配置。没有模型绑定或吞吐收益结论。
