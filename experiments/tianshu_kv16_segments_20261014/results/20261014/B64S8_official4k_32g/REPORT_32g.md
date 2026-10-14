# 天数 BI-V150 32g：B64 八分段正式4K

原版4K输入/1K输出、并发64、每轮256请求，四轮预先排除首轮：**3826.24 total tok/s，765.25 output tok/s，平均TTFT 4457.42 ms**。比上一正式最好3701.31提升 **3.375291%**，比公开基线2028.01提升 **88.669681%**。本配置新跑的Math Level3为 **102/105（97.142857%）**，严格超过95%。

| 轮次 | Total tok/s | Output tok/s | TTFT ms | 秒 | 计分 |
|---:|---:|---:|---:|---:|---|
| 1 | 2293.77 | 458.75 | 7123.21 | 571.43 | 否 |
| 2 | 3662.99 | 732.60 | 4452.26 | 357.83 | 是 |
| 3 | 3915.42 | 783.08 | 4457.35 | 334.76 | 是 |
| 4 | 3900.30 | 780.06 | 4462.66 | 336.06 | 是 |

第2轮3662.99保留在均值中；计分极差/均值 **6.597344%**。短测独立五轮的0.058855%波动不能代替正式测试的稳定性结论。原始CSV聚合共1024成功，每轮1048576输入/262144输出token；这是原始聚合记录核验，未声称正式测试逐请求响应体核验。Git中的CSV为LF副本，归档保留原始字节。

## 改动与证据

相对上一最好，唯一策略改动是B64的adaptive segments4→8。其他批次分段不变；guarded KV16/warp1用于B8/16/32/40/48/56/64，B24等保留KV32/warp2，小批次保留warp4。producer和reducer的tile与分段成对核验。Prefill保持M256/KV128/warp16/stage1/L0，预算2048、threshold0、TopP16、75点预热、gather不变，整数地址偏移64位、BF16，无量化或投机。KV页长仍为16。

新Math105份预测与评分、105自然结束、零请求错误已完整核验；错误index18/39/87。它与上一S4配置的102/105是独立实验，不能混用。Math使用官方batch8、temperature1、top_p0.95、max_tokens32768；C8问答不代表105题均覆盖B64。B64数值与真实C64服务证据另存。五轮小规模确认3928.414099→4065.538627（+3.490582%），只作为进入Math/正式测试的筛选证据。

原benchmark SHA256 `0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`，保持原件。正式5份冻结源码、Math5份命名空间快照、实际CLI、B64S8路由、reducer、原生Prefill和75点预热均通过独立离线复核。原包与分片SHA已核验：

- 正式：3,059,779 bytes/12片，`6428bd042e5b2a9c98d8dd185cb42ad6491b67f19f251c2fd1a3cae99223f813`。
- 新Math：3,838,667 bytes/15片，`9af30208d1de2bee47dded6411a1837959ffa472658fc8ea7f0558008290bf6d`。
- 串行控制器：2,788,448 bytes/11片，`a53da506de183fb680195e1cd5e59bfab536774eb8efd48cef9f12295c4d0ab0`；其3份冻结源、Math105题只读复核及正式子任务成功结束已复核。

完整原始目录位于工作区`service-results/20261014_tianshu`的`kv16_segments_B64S8_official4k`、`kv16_segments_B64S8_math`和`kv16_segments_B64S8_pipeline`。

## 采用与复现

采用为新的4K最好配置，标签`tianshu-best-kv16-b64s8-20261014`。保留旧S4标签及启动脚本。新启动：

```bash
bash experiments/tianshu_kv16_segments_20261014/serve_tianshu_minicpm.sh
```

最终启动脚本SHA `cb995f265bb17e66a4c04bf5823ac513173adf47e5ddf1ce559b6451658a7bf9`。本地与容器使用隔离模拟vllm核验实际Math/正式argv相等、12个正式和10个Math关键开关、插件及compact开关的冻结AST、3份hook源码，均通过；只修改成绩注释，未额外启动GPU。

```bash
python3 experiments/tianshu_kv16_segments_20261014/verify_downloaded_official.py \
  --root /path/to/20261014_kv16_segments_B64S8_official4k_tianshu \
  --math-root /path/to/20261014_kv16_segments_B64S8_math_tianshu
```

此配置未测正式16K。后续页长32仅为独立候选，不能将其性能或正确率关联到本标签。继续按用户要求做4K短测、逐轮报成绩。没有修改沐曦仓库，也没有推送GitHub。
