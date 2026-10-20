# BI-V150 32G：线性INT8值缓存，准备阶段

## 当前证据

正式最好仍是K8V16/default4096：4K 5101.45 total tok/s、1020.29 output tok/s、Mean TTFT4164.20ms；同源码Math Level3 102/105。最佳标签tianshu-original-command-4k-k8v16-budget4096-20261019，提交5ea64d33e594fc9510f5a3062bbf84811161ba5f。

现有layout_spec.py已证明42层、block16、KV heads2、head128、真实UINT8存储及12288字节物理页。K占4096字节、V仍以同storage的BF16视图占8192字节；value_planes.py并没有复制整份V。现有dense_prefill.py只解码K，V直接复制。此前软件FP8同时压缩K/V更慢，不能把页大小减少当成吞吐收益。

沐曦10af519dc5c0的新收益主要来自由排序回退切回原生TopK/TopP；天数当前compose_top_p与top_p_sitecustomize.py已经接入原生TopP，并使用严格全局归一化与几何中点搜索，因此不将该沐曦改动当作天数的新收益。

## 本轮假设

保留K的已验收软件8位编码，单独将V替换为线性INT8。候选只需符号扩展和常量缩放，可能比软件FP8值解码便宜。物理页目标12288→8192字节，减少三分之一页存储；这只是布局计算，不是性能预测。

暂定V缩放1/32、1/16、1/8，必须报告饱和比例和误差；不根据评测题号选择缩放，不改模型权重、temperature/top_p、请求输出、benchmark统计。原最佳版本与其历史源码保持不变，候选使用独立目录与回退路径。

## 实验门槛

1. 先验证256个码点、写入舍入/饱和、负slot、尾部保护、实际页stride和storage边界。K编解码与原算法不变。
2. 真正decode kernel必须读取INT8 V并缩放；dense prefill与容量fallback也必须正确解码V。禁止悄悄保留完整BF16 V副本。
3. 同Graph改变Q/K/V、block table和序列长度后复查，确认没有缓存旧数据。保持有界scratch、真实内存profile计入和capture前分配要求。
4. 组件对照采用同一进程内已验收K8V16，覆盖B1/8/64、4K及5K decode、ragged/mixed/prefill；计时前独立参考检查通过，交替顺序计时并保留原始event与编译参数。不使用历史BF16组件作为新服务提升对照。
5. 只有组件通过且有实质收益，才进行C64/N64的4K/1K服务筛选。服务稳态提升至少2%才晋级；首轮单列、逐轮报告，不常规跑16K。
6. 拟采用版本必须新跑同源码Math，正确率严格大于95%，再跑原脚本N256正式4K；旧Math成绩不能借用。任何失败都记录，不以缩小断言伪造通过。

## 状态与保存

2026-10-20准备时，SSH命令及公钥指纹一致，但网关仍返回Permission denied(publickey)；用户确认已开机且网页IDE可打开。当前只完成源代码审查，尚未实现或运行本候选，无数值、吞吐或Math结论。待SSH恢复后直接在容器实现和验证。

原始实验数据留/workspace/benchmark-results，代码/报告/关键结果推天数GitHub，不建立本地项目副本；不修改沐曦仓库。预约为北京时间2026-10-20 06:00至2026-10-21 00:00，GPU硬停止时间为23:58。
