# 天数32G：K8V16预算4096独立质量与正式验证

只在已完成3+3服务的稳态增益>=2%、候选两计分轮均超过控制且各自spread<1%后运行。保持同一源与41环境；显式serve预算4096/warm83，不能标作原始serve argv不变。

全新官方Math105、batch8、temperature1/top_p.95/max_tokens32768，无错误/无缺题、严格>95%，每题prediction/review/stop reason保存；之前100/105仅用于CPU核验既有解析器，不借给新配置。只有新Math通过才运行原benchmark4K C64/N256四轮，预排首轮，原benchmark源不改。

Math与正式源/argv/env逐项一致，原始CSV重算、KV真实页/41env/预算4096和83warm均查，硬停15:58Z，无16K。当前原命令best4997.05独立保留，不预先采用候选。硬件profile未采集。
