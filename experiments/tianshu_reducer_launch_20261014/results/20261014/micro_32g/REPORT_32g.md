# 天数BI-V150 32g：分段合并warp独立扫描收益偏小

固定当前.cv/S8/guarded KV16/w1 producer、相同grid/tile和原生源码，仅扫描reduce_segments launch的warp1/2/4/8，对照为未传num_warps的默认值。真实编译metadata确认默认warp4/stage3，显式4是身份控制；producer配置和.cv源8deb保持。

|batch|KV最大长度|warp1耗时变化|warp2|显式4|warp8|
|---:|---:|---:|---:|---:|---:|
|8|4096|+0.3881%|+0.0378%|-0.1178%|+0.6003%|
|16|4096|-0.7043%|-0.8083%|-1.0709%|-0.1262%|
|24|4096|-0.1719%|+0.3277%|+0.4170%|+0.3965%|
|32|4096|-0.6752%|-0.4518%|-0.2126%|+0.2833%|
|64|4096|-0.4818%|-0.4117%|+0.0195%|+0.8671%|
|64|5120|-0.4057%|-0.3792%|+0.0400%|+0.7254%|
|64|65|-9.4761%|-7.9475%|+0.0197%|+16.1419%|

B64/4K warp1仅减时0.4818%，5K减时0.4057%；warp2类似。短KV65可减时9.4761%，属于另一个负载，不能外推比赛长上下文收益。当前不足以进入服务实验，不采用、不新跑Math或完整官方。计时包含producer+reducer完整Attention，不声称是reducer自身加速比例。

全部17场景/68候选FP32和选定CPU独立参考、改变Q/K/V后的Graph重放与尾部检查通过，原始765个event重算、6源/实际cv生成源/原L0/原native/编译器快照及producer1组+reducer5组编译产物SHA已下载核验。部分warp1/2/8因归约布局改变而不逐位相同，未将容差正确写成bitwise；显式warp4身份控制逐位一致。未声称新Math正确率。

归档3590054bytes/14parts SHA256 `8786bb4bec54271d3a08aab8a9780939829d8f1ad75f26638a31912475b5d570`。原件service-results/20261014_tianshu/reducer_launch/20261014_reducer_launch_tianshu/micro。复核命令verify_downloaded.py --root <raw>/micro。当前正式最佳4091.29/Math103保持，安装源未改、GPU串行，无16K服务benchmark。
