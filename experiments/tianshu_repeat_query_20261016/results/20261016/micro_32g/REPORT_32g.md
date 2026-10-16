# 天数 BI-V150：BM16重复单query数值正确，但C64更慢

2026-10-16，源f2a2ad9e14e3e19d7b34c04d5ce321020f39dc7c。保持BM16矩阵布局，只把3D单token decode的有效query重复到原本无效的另外8行，用于Q load、score位置和计算掩码；最终store mask与offset、完整epilogue/reducer逐字相同。保留.cv/L0/KV16-w1（B24回退32-w2）、B64S8、64位整数/物理指针和相同grid。False生成副本为恒等控制。

CPU1152映射、native wrapper仅maxQ<=1时3D的源码契约及生成源码精确逆变换通过。GPU17场景/34候选全部通过FP32、选定CPU、原输入逐位、变Q/K/V Graph逐位及输出/scratch尾部保护；459原始event、10份源快照、实际编译metadata及3组B64产物SHA都已下载独立复核。通过数值检查不能保证性能收益。

|batch|KV长度|原版ms|重复query ms|变化|
|---:|---:|---:|---:|---:|
|8|4096|0.140441|0.142401|+1.39522%|
|16|4096|0.264860|0.267627|+1.04447%|
|24|4096|0.407652|0.405664|-0.48760%|
|32|4096|0.502821|0.512976|+2.01961%|
|64|4096|0.845974|0.859610|+1.61180%|
|64|5120|1.052844|1.069113|+1.54521%|

C64主负载明确更慢，不接入服务、不跑本候选新Math或正式benchmark。B24小幅局部变化不外推整模型。真实B64三个版本均85寄存器/0spill，不据此推断occupancy或底层瓶颈；没有硬件性能计数。旧BM8负结果与本次BM16结果独立记录。

GPU controller35426、child35428均终态。归档4021519bytes/16parts，整包SHA256 `3fdb02d8f23743e1c1d6cbed33ff769080fd698ab5a56199da21eff1eaa4063d`。本机原件service-results/20261016_tianshu/repeat_query/20261016_repeat_query_tianshu，分片及整包SHA已通过。

```bash
python3 experiments/tianshu_repeat_query_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/repeat_query/20261016_repeat_query_tianshu
```

上述本机复核已通过。正式原始命令4K3992.97 total tok/s、Math103/105与最佳启动器不变。下一方向单独准备当前原始命令CV下的M2048 GEMM+strictTopP16K组合，旧服务条件不同且无组合证据，不把旧小收益相加。
