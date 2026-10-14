# 天数BI-V150 32g：修复TopP16K tile与maxseq64组合服务结果

4K输入/1K输出、C64、N64，每组新服务3轮，预排除各自首轮。control当前采用.cv/预算4096/83预热点/S8/KV16与nativePrefill，原生TopP8192/4096/w16/s1、默认maxseq256。candidate纯TopP严格修复16384/8192/w16/s1加maxseq64；TopK及不支持输入原生回退。采样1/.95、模型长度131072不变。组合结果不能归因单项。

|组别|第1轮（排除）|第2轮|第3轮|后两轮均值 total tok/s|
|---|---:|---:|---:|---:|
|当前配置|3808.40|4067.23|4052.19|4059.71|
|修复TopP16K +maxseq64|3719.13|4099.41|4104.02|4101.71|

完整复核后的稳态提升+1.034509%，未达到预定2%门槛，不采用，不启动5+5、新Math或正式benchmark。当前正式最佳4091.29保持；本次4101.71为N64短测，不与N256正式成绩直接比较。独立TopP约23%～27%耗时下降未转为同等模型吞吐改善。

全部384请求的4096/1024用量、随机种子、文本、TTFT、wall time与聚合吞吐本地重算；19份冻结源、6份安装源、双方83点预热、实际.cv producer/reducer、maxseq256/64、生成修复TopP源码SHA与纯TopP/TopK路由逐一复核。服务没有新质量结论；原生基线与修复后的严格概率mask存在差别，必须新Math才可采用。模型profile未采集。

归档5215196bytes/20parts，SHA256 6c3ee0ae1643d272fe6bae5e5b2394ede13343ee21e9b4764689a031cacebb00。原始记录位于service-results/20261014_tianshu/topp_precision_maxseq64_service/20261014_topp_precision_maxseq64_service_tianshu。

离线核验脚本继承修改时误把B64 TILE_SIZE断言16写为19，首次本地核验正确拒绝；已仅修正离线断言为实际16并重新完整通过，运行冻结源码与原始观测未变。

下一实验：同一严格修复主体下，独立扫描32K/64K tile/warps，保持十种输入、翻转Graph和1e-6 CPU门槛。
