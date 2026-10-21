# 2026-10-21 21:17北京：逐M融合正式晋升

正式四轮5531.99（排除）/5690.21/5690.13/5707.50 total tok/s，1024请求全成功。
新推荐热均5695.95 total /1139.19 output、TTFT3663.77ms；较5618.01 +1.387324%，TTFT -4.476386%；热轮差幅.304954%。
公开2028.01基线+180.864%（2.80864倍），0抢占、就绪后日志无autotune记录。原始服务/runtime/benchmark/质量审核与独立CPU复核均通过。
Math8完整103/105由独立CPU逐条复核但原GPU覆盖失败保持audit_failed；C64新题101/105，真实M4096×42融合、旧M64图重放/LM64 INT8证据通过。
同471模型源/62环境/13官方serve参数，现有450源逐字节保持；不新增packed权重，M256明确排除。没有例行重测16K。

独立结果：tianshu_m_range_prefill_official_results_20261021。
Bash新推荐入口：tianshu_m_range_prefill_release_20261021/serve_tianshu_minicpm.sh，实际qualification分支CPU干跑通过，未启动第二个服务。
标签：tianshu-original-command-4k-m-range-fused-20261021。旧5618.01 tag保留。

接下来GPU已串行开始BF16大M1008项，runner288549；完成后QKV NN34项runner290252，再O/Down NN各34项runner293298。组件均须完整CPU整数/fixture核验后才能宣称收益。
代码、报告、关键结果放容器和天数GitHub；本机仅临时SSH socket/FIFO通信。
沐曦参考只读已有ed3bcfe/8eb57f17；更新vllm参考副本fetch被自动审批的只读限制拦截，未重试该操作，不影响天数工作。
预约到2026-10-22 00:00，北京23:58 GPU硬停。继续按实际剩余时间找优化。
