# 未来输出KV预留：4K完成，16K定向筛选中

4K/1K C64 N64，双方各3轮、首轮排除，5048.51→5031.56（-0.33585%），384/384成功，没有触发准入拦截，组内/组间固定seed一致。完整GPU服务与计划释放审计通过；详见REPORT_4K_32g.md和260来源的DIAGNOSTIC_4K_GATE_32g.json。

完成4K后补查并修复非法零token调用的原生ValueError转发。旧失败与修复证据均保留；CPU容量检查重新通过。16K使用261来源、52项环境的新版本，针对未来容量准入与抢占做小规模对照，未运行完整benchmark、新Math或正式采用。原件`/workspace/benchmark-results/20261020_future_kv_service16k_v2_tianshu`；控制器run_16k_screen_v2.sh、分析器verify_screen_v2.py。所有服务23:58北京时间硬截止。

这是明确针对KV紧张长上下文的补测；普通候选仍只做4K筛选。跨配置固定seed文本作为诊断，组内重复必须稳定；同最终源码新Math严格>95%并通过正式验证之前始终adopted=false。
