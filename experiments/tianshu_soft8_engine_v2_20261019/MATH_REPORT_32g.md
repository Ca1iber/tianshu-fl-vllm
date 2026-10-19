# 软件8位KV：新Math105质量通过

来源提交9c0d9ed352b2a42755fe950fee071091fc5ce5fc，同一已筛选4K +3.1537%的UInt8 KV profile；模型/Q/out BF16、固定软件E4M3缓存。完整新运行105/105成功、0错误、0 incomplete，**102/105 = 97.142857%**，通过官方>95%门槛。159冻结源、真实42层UInt8 spec/view、内存预算、采样超参/原argv以及全部105预测、105 review和汇总指标已独立本地重算。

错误索引[18, 87, 72]；stop原因{'stop': 104, 'max_tokens': 1}；max_tokens索引[83]。命中上限的样本仍有正常预测并被评为正确，并非API错误。评测保持temperature1/top_p.95/max_tokens32768、batch8、timeout3600。没有复用旧102/105；相同正确数不表示每条输出相同。

下一步正式脚本4K/C64/N256、4轮首轮排除，不运行16K；正式成绩尚未产生，canonical最佳4093.81未替换。

原始包SHA256 `aed08091d275dd9a885df779d09a4ce71d5310b9116dd42bc72db2d3730d586f`，6286539 bytes，24分块与整包已核验。
