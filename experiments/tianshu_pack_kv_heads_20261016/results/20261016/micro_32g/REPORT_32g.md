# 天数：双KV-head合并数值通过，4K最快仍慢11%

源a127a192d07b41de83c7f974aca74f92ed366d65。相同序列的两组KV head合到BM16 query行、每次逻辑16 token对应32列，softmax前屏蔽跨head分数；原序列解析、BF16 dot/FP32 acc、int64、L0/.cv、KV16/S8、epilogue/reducer函数体保留，head grid轴2→1。没有减少完整dense矩阵乘法的理论乘加数：跨head区域也参与dot后再屏蔽，不把填零行复用描述成计算量减半。

4场景16候选均通过GPU FP32、CPU抽点、原与变Q/K/V Graph、尾部检查，无编译错误。180原始event、10源码（含native helpers）、生成源码与5组实际编译产物SHA均下载复核。未扩大完整边界、服务、Math或正式。

B64/4K基线0.846528ms；合并warp1为1.044193ms（+23.350%），warp2为0.939891ms（+11.029%），warp4为1.027302ms（+21.355%）。5K最快warp2慢11.051%；B64/65最快也慢1.827%，B8/33最快慢0.621%。没有主负载收益，不采用。

B64原版与identity寄存器85/85，合并warp1/2/4为178/97/96，均0spill；只陈述静态资源事实，不由此单独推断硬件瓶颈。正式最佳3992.97/Math103/105保持。

原件 ../service-results/20261016_tianshu/pack_kv_heads/20261016_pack_kv_heads_tianshu，4227502bytes/17parts，SHA256 `0f6056fc84780e10e4e9fd03df8f1c2f2710f7f7eb3fc72149ce5a0bf3fd0c02`。runner43754、child43756已终态且hardware_after无GPU进程。

```bash
python3 experiments/tianshu_pack_kv_heads_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/pack_kv_heads/20261016_pack_kv_heads_tianshu
```
