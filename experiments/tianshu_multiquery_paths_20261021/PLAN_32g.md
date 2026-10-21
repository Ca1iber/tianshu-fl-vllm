# 多查询真实路径对照与 T16 候选

当前原26场景在短K控制处failed，19长K独立复核与6短K独立诊断分开保存。短K Q2/K2、Q8/K8的CPU native BF16-P、控制和3候选逐位一致，同时保留同样的理想FP32超限，不删场景/不扩大阈值。新实验改进实际比较对象和tile，不能覆盖原失败结果。

控制使用当前实际SmallArena policy安装的8192pages BoundedDensePrefill和原gather_decode，随后真实native zero-initialization BF16 reader BM256/Q32/T128/w16；无法容纳时走原S16 byte2D。Graph计时包含gather，不把提前解码好的BF16缓存当作免费路径。非因果额外fixture走byte2D仅作独立算术参考，407源目前根本不支持draft，不能冒充现有draft模型路径。候选共享原bytecodec与数值JIT，以原Q4/T32/w2为同场3D对照，另测Q2/T16/w1、Q4/T16/w2、Q8/T16/w2，S沿用相同N策略。

沿用完整26场景矩阵：n1/8/32/64，Q2/3/4/8，4K/5K、混合参差/stride，5种短K、Q7非因果。每项两相位改变Q/K/V、实际长度、页表，真实同Graph重放。长K仍对理想FP32全输出使用原.002+.01*abs阈值；短K额外对独立CPU模拟原BF16-P语义核验，理想FP32误差及失败元素完整记录，不能混淆两种参考。独立短K诊断已解释为何理想FP32本身不能替代原生BF16算法语义。无扩大阈值、删失败case或重写旧门禁。

完整KV/输入/元数据/RNG与边界guards，CPU原K码值及选中注意力向量，实际current route与source/JIT/capture记录；7轮旋转次序各20热Graph。仅单层attention+当前控制必要的gather，不含writer、模型GEMM/sampling/draft/rejection，不是服务吞吐/Math。当前非Spec真实4K最短query128，没有证据该Q<=8路径会提高当前4K；不跑无命中的NoSpec服务A/B。若组件可用，后续新Spec profile需显式处理47层(42causal+5noncausal)KV、loader内INT8权重计费、真实capture前激活、全部draft实际shape预热及正确FP32标准拒绝，现407源guard不可简单删除。

页表分配width+9物理padding保留poison，原大tile尾部最多额外7项ID可安全读，KVload mask原样。所有缓冲实际query token容量+guard。硬截止2026-10-21T15:58Z，单组件40分钟。GitHub仅代码/报告/关键结果，完整PT与日志留容器。

比较范围补充：baseline标签bounded_gather_or_byte表示原生暂存/回退组件。全prefill Q>1的目标uniform场景及over-capacity直接回退与当前路径吻合；混合Q且能暂存时现生产路由会拆leading decode，因此本实验该类fixture只作gather-all算术参考，不冒充真实混合服务成本。每case显式标记baseline_matches_uniform_or_fallback_path，性能结论仅使用该标记true的场景。所有26场景仍检查数值与边界，未删测试。
