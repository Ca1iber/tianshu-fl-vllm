# TopP自适应outlier cutoff：严格通过，bias2.5进入服务筛选

控制为已验证geo32K/16K-trunc/w16/s1，不再使用旧8192版本。候选只把纯TopP初始sigma改为max(原sigma,std_logit-bias)，bias为3.0、2.5或2.0。收集候选质量不足时保留原完整范围回退；最终搜索与重复值处理、p=.95、32次上限、1e-12与CPU严格1e-6门槛不变。正常分布假设仅帮助选起始cutoff，没有替代最终精确判断；TopK保持原分支及kernel。

10场景涵盖B8/32/64正态、C64尖峰、屏蔽/重复值、B32 TopK100、BF16往返正态/尖峰及全相等。每例使用原始输入与翻转后的变化输入，全部三个候选strictCPU语义与动态输入Graph均通过。3520条行语义和350个计时记录已下载重新计算；不改CPU容差，原始控制的语义记录也保留。

|B64输入|geo控制 ms|bias3.0 ms|bias2.5 ms|bias2.0 ms|
|---|---:|---:|---:|---:|
|normal|2.579814|2.586538|2.592635|2.603077|
|peaked|1.687744|1.457917|1.252588|1.516792|
|bf16_normal|2.478369|2.480912|2.474630|2.501014|
|bf16_peaked|1.896377|1.785512|1.672245|1.767525|

bias2.5两种尖峰更快且普通输入基本持平，选它进入下一步服务筛选。计时为包含相同输入restore的Graph区间，7次交替顺序、每次10replay，不能直接外推完整模型吞吐。未记录每行候选数量和搜索轮数，因此数据量减少仍是机制假设，不把它当已测profile事实。

原件 `service-results/20261016_tianshu/topp_adaptive/20261016_topp_adaptive_tianshu`，5221035bytes/20parts，SHA256 `6075b13b9913521cde54eadee859d6e86c1e195daed758974b731b5f441f4ebd`。分片/整包、11源、4份生成源可逆构造及TopK AST、编译产物、所有数值/计时已本机独立核验。运行源码c848273的11份微测源未改。

```bash
python3 experiments/tianshu_topp_adaptive_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/topp_adaptive/20261016_topp_adaptive_tianshu/micro
```

下一步原始官方serve argv、4K/1K/C64/N64、3+3不同种子配对、排除首轮。两组共同保留现best的GEMM三投影、geo32K、CV/KV16/S8、75warm、默认2048/256/asyncTrue和原RNG，只改initial-cutoff开关。环境39项且真实新TopP源码及三个GEMM配置将验证；新Math/正式尚未运行，当前正式最佳仍4093.81、Math102/105。
