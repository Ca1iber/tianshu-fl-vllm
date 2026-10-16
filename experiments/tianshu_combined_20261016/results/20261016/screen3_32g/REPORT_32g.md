# 天数 BI-V150 32g：五项组合筛选提升2.47%，待独立复核

新鲜3+3服务对照，4K输入/1K输出、C64/N64，每组预排首轮。控制沿当前CV/KV16/w1/S8、budget4096/83warm、maxseq256、原生TopP8192/4096、原采样、stream1；候选将预算增至16384/131warm、maxseq64、严格TopP16384/8192、融合采样、stream4。两边async=True、BF16、native Prefill及相同attention源码。16384不是16K输入benchmark。

|组别|轮次|total tok/s|output tok/s|TTFT ms|
|---|---:|---:|---:|---:|
|control|1（排除）|3804.43|760.89|17396.97|
|control|2|4086.86|817.37|11750.21|
|control|3|4033.42|806.68|11777.18|
|candidate|1（排除）|3832.08|766.42|17067.05|
|candidate|2|4161.70|832.34|12401.75|
|candidate|3|4159.04|831.81|12419.98|

稳态4060.1406→4160.3683 total tok/s，提升 **2.468579%**。候选后两轮均高于控制后两轮，但平均TTFT增加约 **5.50%**，需保留这一取舍。不能把N64筛选当正式N256成绩，不能把单项收益相加。

全部384条响应及用量/正文/逐请求TTFT和聚合数值本地复核通过。23冻结源码和源码快照、安装采样两源及严格TopP重新渲染字节、实际83/131预热、真实maxseq256/64和async=True、CV/KV16/S8 producer/reducer、B64融合采样、实际流式命令、KV容量>=327680和全部轮次零preemption均通过。额外保存两边API进程实际/proc环境白名单（39个非秘密profile变量），供后续完整复现。

控制三轮均无新增调优记录。候选首轮有2条运行时索引调优记录，后两轮均0；完整记录见runtime_tuning_records.json，不据此声称全部编译成本已被消除。

归档5384251bytes/21parts，SHA256 `75c55537008750964bd6ed47721d083d5790bf57bab8a13f75610539c87186f8`。本机raw：`service-results/20261016_tianshu/combined_screen_service/20261016_combined_screen_service_tianshu`。复核：`python3 experiments/tianshu_combined_20261016/verify_downloaded.py --root <raw-root>`。

已达到预设>=2%且稳态不交叠门槛，独立fresh5+5确认（预排前两轮）已串行启动。尚无该组合的新Math或正式benchmark，不采用为新最佳；正式最佳仍4091.29 total tok/s、Math103/105。
