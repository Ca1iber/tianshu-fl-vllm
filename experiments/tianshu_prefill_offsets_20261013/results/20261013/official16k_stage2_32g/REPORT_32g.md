# 天数32g：Prefill stage2+Decode32正式16K

原始16K/1K C64 N128、四轮排除首轮：2665.42 total tok/s、156.79 output tok/s、TTFT184434.62ms。比上一套同Decode32/Prefillstage1的2600.48提升2.497231%，TTFT变化-2.911155%。比今天此前2440.62累计吞吐提升9.210774%。

|轮次|总tok/s|输出tok/s|TTFT ms|duration s|计分|
|---:|---:|---:|---:|---:|---|
|1|2410.37|141.79|193361.51|924.43|否|
|2|2669.43|157.03|184141.35|834.72|是|
|3|2664.26|156.72|184527.41|836.34|是|
|4|2662.58|156.62|184635.11|836.87|是|

四轮512/512成功，每轮总2097152输入/131072输出，0失败。后三轮波动0.256995%。对应新的独立Math103/105、98.095238%、0执行errors，未复用旧102/105。数学与服务A/B及正式source SHA完全相同，实际Prefill M256/KV128/w16/stage2记录齐全；Decode32 c407与Stage1基础配置、Native L0 kernel4a1、75桶/固定gather/budget2048保留。

本增量仅Prefill pipeline1→2，核函数源body和网格未变，浮点输出/指针类型不改。既有旧stage1自动报告模板会描述基础配置，实际stage2_sources/config2及此报告为真实最终profile。没有混入SQL反射或KV写入原型。

原始归档2348350bytes、9分片与整包SHA256 0b35cf62b7315f49e7eca41bee9a666dfce0aecbcfa55e96c336b4c17b1d1d45全部本地verified。目录service-results/20261013_tianshu/prefill_stage2_official16k/official16384，完整四轮响应/CSV/源/质量匹配证据保留。Git CSV为LF副本，原始归档原字节不改。原benchmark SHA0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565未修改。

同一新profile正式4K正在顺序运行，最终统一profile/default选择等两项都闭合后更新。历史4K3222.52仍保留，未把不同profile的最好成绩合成一次运行。
