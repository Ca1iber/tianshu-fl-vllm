# 天数32g：KV页长16与32的组件检查

18个Decode与9个真实KV写入/Prefill/混合用例完成，GPU FP32及CPU抽点参考、padding/segment尾部、页表置换和stride检查通过。Decode页长两组输出逐位相等；Prefill每例两个in-place修改输入的Graph重放相等，KV writer整个物理存储与期望逐位相等，包含slot=-1跳过。实际builder/writer/forward及原生M256/KV128/w16/s1/L0组合执行通过。

独立本地复核全部记录的参考指标、尾部标志、9或7个原始设备事件时间的中位数与百分比；Prefill14份冻结源码快照与工作区自有源码核对通过。指标是GPU执行时保存的标量；未序列化全部GPU Tensor。Decode探针及L0生成源也保存在原包，可按确定性seed重跑。

页长32在4K Decode主要批次慢2.4%至3.35%，B64慢2.66%；Prefill2048快6.91%、chunk2048快7.69%、mixed64快6.81%、mixed_ragged快10.77%。这些是组件时间变化，不是服务吞吐结论；仅据此不采用。B64S8为双方基础，页长仍需同期N64三轮服务对照、新Math及正式测试。

Decode原包2,826,149bytes/11片，SHA `2f62caa2a1442e9228a4a1ed0e5efcd5c32e2ac2159d015bf524720267640bca`。Prefill原包2,895,707bytes/12片，SHA `153419ef71f03af1e696aa398ed81d341f8481ce2b46443141faf90b962f2913`；分片及全包SHA已校验。原件位于工作区service-results/20261014_tianshu/kv_page32_decode与kv_page32_prefill。

进程内guard仅扩展页长16/32，其余原supported判断保留，并串接已验证的NativePrefillLoader；未改安装文件。没有16K服务测试、沐曦修改或推送。
