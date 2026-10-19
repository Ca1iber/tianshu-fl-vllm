# 天数 BI-V150 32G：K8V16 scheduler budget 2048/4096

独立短测，非正式分数。两侧均已验证 K8V16、GEO TopP、M2048 tiles、75基础点和99可扩展点。只对candidate显式增加 --max-num-batched-tokens4096；实际warm自动75→83，明确不称原始serve argv不变。当前best启动器、安装源和原benchmark不改。

原因：KV流量减少后瓶颈可能迁移，重新测现有框架的每步token预算，避免将旧BF16收益直接套用。3+3全新服务，4K/1K、C64/N64，各剔除首轮。保存384响应、canary、实际worker预算、75/83预热、页分配、算子配置与所有源/env，不改变采样算法或模型。

CPU只验证真实argv组装与重复参数/输入不变守卫；实际框架行为由GPU服务的worker与warm记录检查。小规模有明确收益后才新鲜Math105>95%与原benchmark4K；未测16K。当前原命令正式best4997.05及其Math100/105不归属本候选。硬件profile待测。
