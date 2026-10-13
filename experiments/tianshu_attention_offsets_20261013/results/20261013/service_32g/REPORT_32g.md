# 天数 BI-V150 32g：Decode int32 元素偏移服务对照

完整16K/1K、C64/N64服务小测排除首轮后：控制 2393.194947 →候选 2562.170850 total tok/s，提升 7.060683%。候选末两轮差距 0.118906%，且两个计分轮都高于控制组最高计分轮。不是原始N128正式16K成绩，未更新正式best。

| 轮次 | 控制 total tok/s | int32 total tok/s |
|---:|---:|---:|
| 1 | 2136.855008 | 2252.282476 |
| 2 | 2375.548109 | 2563.694134 |
| 3 | 2410.841785 | 2560.647565 |

控制组复用首次尝试中完整且未改动的三轮控制数据；第一次候选因BF16 K/V descale守卫过严未实际路由，已停止且不计比较。修正后的候选新服务单独重跑三轮。两个工作负载、提示seed、installed sources、decode/tile/bucket hooks完全一致；实际run_service_trial.py与控制快照逐字匹配。只在mode NONE/BF16/B4..48且偏移上界<2^31时选择int32，物理指针和浮点计算保留原样，B1/B64等回退原64。

两组共384个吞吐请求全部成功，usage均16384输入/1024输出，7组稀有shape greedy响应相同。候选记录30个batch/stride/segment组合，包含实际K/V descales非空，mode NONE和安全上界；生成内核SHA c4072580f4274f89ee509c1eda1929d9eb5d74b2209a8c339534e543679fbf66。真实参数10场景GPU FP32/逐位一致先行通过。

有效候选完整归档3564247 bytes、14分片及整包SHA256 d29139782d1d1659b5078233a898bd674c55729a76aeaf45f6128d806f9abca8全部本地verified。原控制/失败候选原始归档1705885 bytes，SHA256 ed1090e2c9e1089ef8138d0f8cdabda33d96503dbf0b1144c7a427aef9ba1282同样verified。本地数据分别在 service-results/20261013_tianshu/attention_address32_service_v2 和 attention_address32_service_stopped_v1。

目前证明的是本地短测服务收益。新匹配的独立Math和正式benchmark尚未完成，不复用旧102/105来认证新计算路径。后续顺序控制器继续GPU独立实验与质量评测，只有新正确率>95%才进入原始正式16K和4K，各四轮、排首轮，benchmark源未修改。框架SQL反射只在单独服务测试，不混入本次candidate；KV写入变化也没有混入。正式最好仍4K3222.52/16K2440.62。
