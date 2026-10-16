# 天数：TopP补偿求和单独仍失败；实际最大值加补偿通过

源 fe7959930972d73cf97295b237ab2e0df7474fc6。所有候选采样参数、RNG、CPU FP64独立参考及1e-6概率质量门槛不变；原生TopK走原8K/4K/w16，安装文件不改。

固定10输入、6配置，含B8/B32/B64 normal、B64 peaked、masked/ties、TopK、BF16 normal/peaked和all_equal。原/翻转输入Graph通过后才计时；4096逐行语义记录、406原始event、8冻结源码和3生成源码已本机复核。

原32K以及只用FP32 Neumaier树归约和Kahan跨tile累加的32K均在B64 peaked严格概率质量检查失败，不计该场景时间。实际global max作为exp偏移再加补偿求和则10输入全部严格通过。此证据支持继续分离指数偏移与求和开销，但不能据此断言所有误差只有一个原因。

B64 normal：严格16K 3.514845ms；原32K 2.935881ms（全场景未过）；补偿32K 3.383961ms（全场景未过）；max+comp 3.384019ms（全场景通过）。B64 peaked：严格16K 2.266314ms，max+comp 2.349077ms。只比普通输入略快，尚未接模型服务、新Math或正式成绩。

原件 ../service-results/20261016_tianshu/topp_sum_precision/20261016_topp_sum_precision_tianshu，5000069bytes/20parts，SHA256 `50a320e8773ddb0fbf7509d9bfdb948e82361834f4b3bc85ad824764ba00d359`。runner40116、child40118均终态且hardware_after无GPU进程。

```bash
python3 experiments/tianshu_topp_sum_precision_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/topp_sum_precision/20261016_topp_sum_precision_tianshu/micro
```
