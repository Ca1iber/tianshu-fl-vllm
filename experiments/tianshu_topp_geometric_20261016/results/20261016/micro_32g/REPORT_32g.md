# 天数：TopP几何中点搜索严格通过，B64算子耗时再降12%到19%

运行源码f00139bc221ff0009683d6c8b731832e51ca939d。基于actual-global-max稳定归一化及概率域精确重复值修复，standalone TopP只把两处搜索midpoint改为sqrt(lo)*sqrt(hi)，lo=0退回算术中点。p=.95、32轮上限、1e-12区间停止、CPU FP64 mass/minimality/order 1e-6与RNG保持；TopK函数AST相同且实际回退原生8K/4K/w16。

三个fixed候选在10输入全部严格通过，原与翻转输入Graph重放检查通过。原生/identity保留已知严格语义错误，仅按恒等对照放行。10源码及两份生成源、全部逐行原始语义和7轮event下载重算，见RAW_AUDIT。尚未接模型服务/新Math/正式，不把算子数字当模型成绩。

|B64输入|原生8K ms|actual-max32K ms|几何32K ms|几何16K ms|
|---|---:|---:|---:|---:|
|normal|4.581795|2.944461|2.586270|3.092702|
|peaked|2.817191|2.110414|1.700271|1.776692|
|BF16 normal|4.206495|2.759286|2.465874|2.685041|
|BF16 peaked|3.152501|2.259141|1.897581|1.851795|

B64普通/peaked相对actual-max32K分别减少约12.16%/19.43%耗时。未记录逐行实际迭代次数，不能把耗时差直接称作已测的扫描次数减少。没有修改原安装源，正式最佳3992.97/Math103/105不变。

原件 ../service-results/20261016_tianshu/topp_geometric/20261016_topp_geometric_tianshu，4755911bytes/19parts，SHA256 `bdf83e997951b53f6ddbd87704a274ed49caeb20d87d5ea820f6ff2a6ac07aca`。runner44463、child44465均终态，hardware_after无GPU进程。

```bash
python3 experiments/tianshu_topp_geometric_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/topp_geometric/20261016_topp_geometric_tianshu/micro
```
