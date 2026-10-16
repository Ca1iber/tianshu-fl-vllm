# 天数：在线softmax log2域数值通过，但C64慢约1.5%

源b40d2e0cacbf0045f4a87e3282c9711e34d70e2c，实际cv8deb及helpers59d9固定。3D普通Decode将score scale折叠log2(e)、在线softmax用exp2，分段M写回乘ln(2)后仍用原自然指数reducer；不改int64、物理指针、.cv/L0、tile/warp/stage/grid/分段。False恒等对照保留。浮点舍入改变，不声称逐位等价。

CPU35个独立标量online-softmax与自然域分段合并参考通过。GPU21场景42候选通过全量FP32、CPU抽点、原与变Q/K/V Graph及尾部保护；变query×.5、K取负、V×.5，QK分数确实改变。含4K/5K/短边界/stride/padding、peaked query×4/×8、低幅度×.25、全零query。567原始event、11源码快照/当前仓库源、生成源码、helpers来源与3组实际B64产物SHA均下载复核。

C64/4K耗时+1.514513%，5K+1.456097%，peaked4K+1.566101%；没有可用主负载收益，不接服务、新Math或正式。False恒等副本逐位原实现相同；True多数不逐位，但既定atol.02/rtol.01独立参考门槛全过，未放宽。完整逐场景数值和原始事件见RAW_AUDIT_32g.json及原件。

真实B64原版/恒等/候选寄存器分别85/85/88，均0spill。没有采集硬件性能计数，不由寄存器/代码差异推断底层瓶颈。正式4K3992.97 total tok/s、Math103/105与最佳启动器不变。GPU runner38612、child38614已终态。

原始数据service-results/20261016_tianshu/log2_decode/20261016_log2_decode_tianshu，4071749bytes/16parts，SHA256 `d54968abe05fa6c128cecfc734ce8748fcf51389b61b33c35d7b81e2c330c2f6`，分片及整包已校验。实际本机复核通过：

```bash
python3 experiments/tianshu_log2_decode_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/log2_decode/20261016_log2_decode_tianshu
```
