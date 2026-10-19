# 软件8位KV整条组件：decode更快，当前prefill严重退化

启动源14075942f295b4ab9c752fb49c7e7fef638b9af4。V2全部16场景/32改数据Graph阶段完成：实际metadata builder、编码writer、TritonAttentionImpl.forward；完整KV物理存储、output/segment尾部、RNG保持，832个已记录FP32/CPU参考检查通过。65536 GPU编码与256解码/符号零穷举匹配；4个真实Spec/reshape CPU fixture通过。离线完整核验74源、reader/writer可重建、全部编码表/phase/224原始event。不是全模型质量或完整engine allocator证明。

|场景|BF16 ms|Soft8含writer ms|耗时变化|
|---|---:|---:|---:|
|prefill2048|0.713469|22.941411|+3115.48%|
|chunk2048/历史4K|1.841656|65.181873|+3439.31%|
|mixed64|2.890214|68.527344|+2271.01%|
|decode64/4K|0.882103|0.832066|−5.6724%|
|decode64/5K|1.091184|1.027477|−5.8383%|
|decode16|0.271763|0.370233|+36.23%|
|decode24|0.418095|1.375396|+228.97%|
|decode32|0.525281|0.419289|−20.18%|

当前prefill沿原BF16最优BLOCK_M256/TILE128/w16，不能假定它也适合软件解码。已有B64纯decode正收益不足以弥补当前prefill退化；不接服务，不复用102/105，不采用。尚未记录这版prefill实际寄存器/spill，不能把根因定为溢出。下一独立微测扫描15个prefill tile/warp，保持codec、writer、attention算法和B64 decode不变，实际记录资源信息；若无法解决则停止该配置。

量化改变KV值，当前参考针对量化后的自身attention正确性，不等于保持原BF16或证明Math>95%。实际engine spec hook和完整缓存生命周期尚未安装。

完整归档SHA256 `b2c6945764e7bd01df253aac5f3d7e1d9a62988ef8fa80c1e081029b4431e813`，19分片/整包核验。原件service-results/20261019_tianshu/soft8_pipeline_v2_micro/20261019_soft8_pipeline_micro_v2_tianshu。大型GPU Tensor未序列化，离线核验是记录和源码/事件而非重做整个GPU attention。正式最佳4093.81与Math102/105不变。
