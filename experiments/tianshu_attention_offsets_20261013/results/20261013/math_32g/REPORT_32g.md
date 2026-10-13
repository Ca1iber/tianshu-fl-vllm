# 天数32g：地址32计算路径独立 Math-500 Level3

新配置独立重新评测105题，102正确、97.142857%，0 errors，105预测/判分索引完整且唯一，满足accuracy>95%。错误题索引18、70、87。旧profile的102/105没有复用；两个结果分开保存，新候选错误索引也不同。

模型/workload与官方命令一致：MiniCPM5-2B，API端口9031，Level3，batch8，temperature1.0/top_p0.95/max_tokens32768，timeout3600。计算profile保留Native L0 Prefill、adaptive Decode KV16/w4/stage1、budget2048、75 Prefill buckets、fixed logits gather；只在满足安全守卫的Decode批次选择32元素偏移。未混入SQL反射或KV写入改动。

实际32hook和helper逐字匹配此前+7.06%的16K/N64服务候选，9条地址32路由、mode NONE、所有偏移上界<2^31；生成内核SHA c4072580f4274f89ee509c1eda1929d9eb5d74b2209a8c339534e543679fbf66与服务相同。旧Prefill/Decode/buckets/gather源审计均通过，完整预测/评阅/执行日志保留。

原始归档2960797 bytes、12分片与整包SHA256 e952bd750afec47daad5002324882434e4479b9bb526c5b8cfe7b36590cc0b9b全部本地核验。原始目录service-results/20261013_tianshu/attention_address32_math/math。run_math.py实际快照逐字核对；相关原始benchmark没有改动。准确率完成后顺序控制器已启动原始16K/1K/C64/N128四轮、排首轮，再继续原始4K/1K/C64/N256。正式成绩尚未闭合，不提前更新best。
