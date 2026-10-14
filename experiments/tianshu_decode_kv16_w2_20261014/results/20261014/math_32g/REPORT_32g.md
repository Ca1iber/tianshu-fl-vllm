# 天数 BI-V150 32g：KV16 单 warp 新独立 Math-500

新配置官方 Level3 完整105题：**102/105（97.142857%）**，严格高于95%。请求、预测和评分均105项成功，0 API/评分错误；不是复用旧KV32/warp2的103/105。

使用独立五轮确认的相同计算配置：原生 GQA Prefill M256/KV128/warp16/stage1/L0，Decode 对 batch8/16/32/40/48/56/64 使用 KV16/warp1/stage1/L0，其他支持的 batch 保留KV32/warp2或warp4，producer和reducer的tile相同，75点预热、gather、budget2048。物理指针与地址偏移保持原宽度，无投机或量化。

官方参数保留：math_500、本地数据集Level 3，eval-batch-size8，temperature1.0、top_p0.95、max_tokens32768、timeout3600、--ignore-errors。实际server argv与五轮候选完全一致。评测执行约27分钟，单独从新服务启动，未与其他GPU实验重叠。

103题正常停止，2题触及32768-token上限；两题都保留在全部105项评分中，没有删题。错误index为18、87、101；截断index及usage见generation_boundaries_32g.json。

核验四份候选源与当前仓库/归档快照一致，六份安装源快照一致，实际 B8 KV16/warp1/shared12288 路由及相同tile合并、TopP16、75点完整预热和int32 gather均已核验。完整105份预测与reviews、唯一index、原始报告计数/精度、所有raw SHA都在本地重新检查。GPU在此评测结束后已释放，之后由独立串行控制启动正式4K。

全包 3,775,526 bytes，15份分片，全包SHA256：`d76803c9c3dcbb2bcfe9632cc388804ca4a82dff33d9c61caaed502c9a64be8c`。原始路径：工作区 `service-results/20261014_tianshu/decode_kv16_w1_math/20261014_decode_kv16_w1_math_tianshu`。

复核命令：

```bash
python3 experiments/tianshu_decode_kv16_w2_20261014/verify_downloaded_math.py --root /path/to/20261014_decode_kv16_w1_math_tianshu --confirm-root /path/to/20261014_decode_kv16_w1_confirm5_tianshu
```

结论：正确率门槛通过，进入原始脚本4K N256四轮、排首轮的正式测试；正式分数仍待完成和完整备份核验。当前推荐标签及3640.16 total tok/s正式最好保持不变。没有新16K服务评测。
