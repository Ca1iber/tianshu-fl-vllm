# 天数：TopP32K实际最大值归一化严格通过，算子耗时降低

源c5de1c15820ffb5f6a82bcb87b388f276c0af58a。沿用严格概率域mask修复，先求actual global max，再以该值为exp偏移计算归一化，保留普通FP32树归约；TopK原生8K/4K/w16回退，p=.95、RNG、CPU FP64独立参考及1e-6门槛均不变。

10输入6配置全通过原/翻转输入严格语义与Graph；4224逐行语义记录、420原始event、9源码快照和3生成源码下载复核通过。新32K在B64 normal为2.931748ms，原生4.586655ms、严格16K3.498277ms；B64 peaked为2.101757ms，原生2.814857ms、严格16K2.291854ms；BF16 normal为2.758350ms，原生4.204151ms；BF16 peaked为2.259262ms，原生3.154063ms。all_equal新32K0.401449ms比原生0.272453ms慢，完整保留，不声称所有输入均加速。

本轮没有模型服务、Math或正式成绩；准备独立服务对照。原正式最佳3992.97/Math103/105不变。runner40705已终态，hardware_after无GPU进程。

原件 ../service-results/20261016_tianshu/topp_actual_max/20261016_topp_actual_max_tianshu，4905066bytes/19parts，SHA256 `3a959edae8763227dc051c8e1db8b4230f8f5f43951bdc2d85b081a747a9a2e4`。

```bash
python3 experiments/tianshu_topp_actual_max_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/topp_actual_max/20261016_topp_actual_max_tianshu/micro
```
