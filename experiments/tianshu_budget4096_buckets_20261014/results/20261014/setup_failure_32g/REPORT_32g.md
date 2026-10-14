# 天数32g：4096预算首次候选被旧实验入口拒绝

第一次控制组2048完成三轮4043.482097/4068.395611/4066.675380 total tok/s，192计时请求全部合法；实际75点补齐、S8/SQL和扩展hook都通过。随后候选4096尚未开服，被共享greedy_screen foundation的budget必须2048检查拒绝，候选无server.log/state.json/请求。没有有效A/B比较，不能把控制结果当成4096负例或性能收益。

这是复用测试脚本时遗漏旧限制，不是83点策略的GPU数值/内存失败。共享foundation保持原样；修复ea3292c在本实验建立逐字副本，只修改允许预算的守卫及对应错误信息。实际CLI/前置检查执行6例：共享入口只接受2048，自有入口接受2048/4096，8192仍拒绝。该入口固定decode_graph_max0；不把它称为通用任意预算工具。

旧控制数据不复用到修复后的比较。原失败root `20261014_budget4096_after_stages_tianshu` 完整保留，新root `20261014_budget4096_after_stages_v2_tianshu` 重新运行同期3+3。旧寄存器等待器随上游失败正常停止，新等待器跟随v2父进程，不越过失败root。

原包347,179bytes/2片，SHA `73fa9075fd2303b5f60e18031e3417efa51fcd92c795fdb85fab9c081bb03396`；逐片和整包验证。旧父/子全部冻结源码快照、192响应usage/TTFT/吞吐逐一复核，候选未开服及无comparison.json已确认。此为手动失败归档，不伪造complete以绕过通用成功归档器。

复核器`audit_setup_failure.py --root /path/to/20261014_budget4096_after_stages_tianshu`。数据`service-results/20261014_tianshu/budget83_setup_failure/20261014_budget4096_after_stages_tianshu`。当前最好3826.24/Math102不变，无新16K或Math结论。
