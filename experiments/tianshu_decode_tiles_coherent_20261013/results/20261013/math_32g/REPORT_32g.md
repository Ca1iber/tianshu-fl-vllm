# 天数 BI-V150 32g：L0 DecodeKV32独立Math

新配置独立评测官方Math-500 Level3：103/105（98.095238%），满足严格>95%门槛。105项全部完成，0执行错误，所有index唯一且预测/review一一对应；错误index18、87。104项正常stop，1项max_tokens（index101），该截断项评分为正确。

质量配置是Native Prefill L0/stage1、adaptive Decode KV32/w4/stage1、Decode无sink L初值0、producer/reducer TILE32一致、75桶/固定gather、budget2048、BF16，整数偏移/物理指针仍64位。不含Decode32位地址、Prefill stage2、KV writer、SQL反射、投机或量化增量。

EvalScope命令的AST与此前官方评测命令一致：105项Level3，batch8，temperature1/top_p0.95/max_tokens32768。为这个候选新建独立runner/auditor，直接检查实际producer/reducer TILE32及L0生成源，未伪造旧TILE16路由以满足旧审计。13组实际两阶段路由配对、无sink L0生成源SHA4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e，与小规模4K服务和22场景数值证明一致；候选hook SHA及全部6份安装计算源与小测匹配。

新结果不复用旧102/105或另一套Prefill stage2的103/105；单次随机评测不能据此宣称准确率改善。小规模同期4K服务提升6.82%且7项greedy只有6项一致，本次独立质量提供官方门槛证据。正式4K N256已条件启动，尚无完整计分结果，默认最好3222.52保持。

原始归档3210775bytes、13分片，SHA256 55632c9f4b5ba32532507dbafdab57ecba6b30660971cc1d0be9c359470460eb，所有分片及整包、本地source_snapshot、逐项预测/reviews与report内部hash均已核验。目录service-results/20261013_tianshu/decode_coherent_math/math。数据原字节保留；run_math_eval.py源快照与提交文件逐字相同。GPU清理已核验后才启动正式4K。硬件性能剖析未采集，TODO。

安装源码、权重和官方benchmark未改，沐曦只读；用户要求后续以4K优化，此流程没有运行16K服务benchmark。
