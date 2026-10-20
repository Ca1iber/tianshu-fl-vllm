# 新长分段原始双负载正式测试：正在运行

当前新Math102/105（97.142857%）已通过，同一225份源、46项环境、原始serve argv正在执行4K/1K C64 N256和16K/1K C64 N128，各四轮排首轮。当前尚无完整正式均值，不以小规模N64成绩替代正式分数。

archive_official.py为离线归档入口：必须两个正式负载和完整105题Math都结束，重新核验源码及两套快照、真实服务环境、原始benchmark哈希、1536个成功请求、八轮CSV和预设后三轮均值、原生FL Graph真实绑定。它不启动模型、不改变运行配置、不挑选成绩。脚本同时保留原始CSV的SHA与GitHub规范LF副本的SHA、全局抢占计数及服务就绪后的调优日志；不声称现场完全无调优。

当前原件目录`/workspace/benchmark-results/20261020_long_segments_official4k16k_tianshu`；待完成后输出关键JSON/CSV与结果报告，原始服务日志和编译缓存仍留容器。
