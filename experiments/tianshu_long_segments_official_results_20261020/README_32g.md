# 原始4K+16K正式评测完成：5099.65 / 3723.12 total tok/s

新长上下文S16版本已完成全部八轮，1536/1536成功；每个负载四轮排首轮。4K/1K C64 N256计分均值5099.65 total tok/s，16K/1K C64 N128为3723.12 total tok/s。全新Math102/105（97.142857%），相同225份源、46项环境和原始serve argv核验通过。已通过独立采用判定，详见[完整报告](REPORT_32g.md)和[ACCEPTANCE_GATE_32g.json](ACCEPTANCE_GATE_32g.json)。

archive_official.py为离线归档入口：要求两个正式负载和完整105题Math均结束，重新核验源码及两套快照、真实服务环境、原始benchmark哈希、1536个成功请求、八轮CSV和预设后三轮均值、原生FL Graph真实绑定。它不启动模型、不改变运行配置。原始OFFICIAL_GATE/AUDIT中adopted=false快照保留，正式采用另行记录；FOUR_K_COMPLETE_SNAPSHOT_32g.json是4K先结束时的历史阶段快照。

原件目录`/workspace/benchmark-results/20261020_long_segments_official4k16k_tianshu`；GitHub仅保存关键JSON/CSV和报告，原始服务日志、张量和编译缓存留容器。全局24次KV抢占、就绪后一次索引调优等限制见完整报告。
