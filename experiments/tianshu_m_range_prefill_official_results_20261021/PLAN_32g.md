# M范围融合正式4K：独立CPU核验与晋升条件

使用正在运行的20261021_m_range_prefill_resume_official4k_tianshu原始结果，不另起GPU、不改原结果。
原官方benchmark源码/13启动参数/62配置/471源与已通过小测及正确率绑定。
保留官方C8原audit_failed和CPU覆盖复核103/105；真实C64补验101/105且M4096×42层融合通过。
正式N256×4排第一轮：1024成功、完整原始CSV/summary一致；逐轮报告，0preemption、无请求期autotune。
重新执行原runtime与benchmark审核，核对源快照/实际环境/控制器/两轮质量绑定；不只相信green标志。
晋升要求：公开4K吞吐/TTFT门槛满足、热轮差幅<1%、相较5618.01总tok/s提高至少0.5%。
组件或N64小测不是正式成绩；TTFT与旧正式比较也应如实记录。
无论晋升与否均保存代码、报告、关键成绩，完整raw/tensors/回答只留容器。
