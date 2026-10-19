# 软件8位KV叠加TopP grid2：筛选+0.453%，不单独采用

同一软件8位KV实际profile，原serve argv/default2048/256seq/75warm/M2048三投影，唯一开关为TopP grid2。4K/1K、C64/N64，3+3轮首轮排除，稳态control4172.606166→candidate4191.503558 total tok/s，+0.452892%。不据此启动新Math/正式，不替换4259.87正式最佳。TopP此前在BF16下约+0.807%，叠加后并未等量保留。

|模式|轮次|Total tok/s|Mean TTFT ms|新增调优|
|---|---:|---:|---:|---:|
|control|1|3965.029765|16410.19|0|
|control|2|4173.576011|12774.55|0|
|control|3|4171.636321|12777.68|0|
|candidate|1|3988.857294|16453.18|0|
|candidate|2|4195.785725|12754.20|0|
|candidate|3|4187.221391|12872.66|0|

全部384响应、4条within-mode seeded canary、16份零抢占metrics、183源快照与当前仓库源码、实际42环境仅开关差异、42层UInt8 spec/view、已计入内存的共享arena、实际TopP grid/buffer/warp/source路由已本地独立核验。六轮均0新tune，不把环境波动作为未验证解释。两种模式都保持量化KV，改变sampler/RNG并非实验变量；候选没有新质量结论。

原件`service-results/20261019_tianshu/soft8_grid_stack_service/20261019_soft8_grid_stack_service_tianshu`，7831479B/30parts和整包SHA通过，SHA256 `42fc2d068a343622dea60cad102e91e7d1d10934cffa91cb5d15c2412c1c64f2`。启动源55d0bb643aa139cc4a60cabd557ec48858b38656。组件的逐行语义门槛保持retained不变AND（exact baseline OR CPU严格1e-6）；没有放宽。

```bash
python3 experiments/tianshu_soft8_grid_stack_20261019/verify_service_downloaded.py --root ../service-results/20261019_tianshu/soft8_grid_stack_service/20261019_soft8_grid_stack_service_tianshu
```
