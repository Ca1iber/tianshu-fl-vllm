# 天数 BI-V150 32g：B64 八分段新 Math105 完整质量验证

只把已验证guarded KV16/w1配置的B64分段4→8，其余批次保持原策略；B24等保留KV32/w2。新鲜独立官方MATH-500 Level3 **102/105 = 97.142857%**，严格>95%。105份预测及评分完整、唯一索引匹配、0错误、105自然结束、0 max_tokens截断。错误索引[18, 39, 87]全部保留。

这份结果属于新的B64S8候选，没有沿用旧S4的102/105。当前新结果与旧结果正确题数相同，不宣称逐题输出相同或确定性的精度提升。

官方数据集/workspace/evalscope-datasets/math_500、Level3、eval batch8、temperature1/top_p0.95/max_tokens32768、timeout3600、ignore-errors均保持原设定。ignore-errors不能代替无错误检查；本地重新核对105份预测/评分和原始文件SHA。没有修改模型、采样算法、权重精度或评测源码。

5份命名空间实验源码及当前字节、6份安装源码快照、同一独立五轮候选server argv、实际KV16/w1、B24回退、producer/reducer成对tile与segments、原生Prefill M256/KV128/w16/s1/L0、原生TopP16、75点完整预热、int32 gather全部核验。正式运行保持预算2048/long-prefill-threshold0、显存0.85、最大长度131072、FULL_DECODE_ONLY。

B64的绑定证据范围必须区分：新Math并发8，B64S8记录来自启动/捕获及此前真实4K并发64服务验证；不声称官方105题各自使用B64。B64数值与图重放的独立微测另存，官方质量仍完整遵循batch8。

原始包3,838,667bytes/15分片，逐片和全包SHA通过：`9af30208d1de2bee47dded6411a1837959ffa472658fc8ea7f0558008290bf6d`。全部预测、评分、日志、源快照和实际路由原件：`service-results/20261014_tianshu/kv16_segments_B64S8_math/20261014_kv16_segments_B64S8_math_tianshu`。

复核：

```bash
python3 experiments/tianshu_kv16_segments_20261014/verify_downloaded_math.py --root /path/to/20261014_kv16_segments_B64S8_math_tianshu --confirm-root /path/to/20261014_kv16_segments_B64S8_confirm5_tianshu
```

新Math严格通过后，顺序控制156140已启动原版正式4K四轮（runner157705/API157709/benchmark158167），预排首轮。当前检查点尚无正式完成轮或新正式均值；小规模4065.54仅用于筛选。正式最好仍3701.31，只有新正式成绩实测超过旧最好并完成原始包/源码/质量/启动器核验才更新tag。未跑16K，未修改沐曦仓库或推送。
