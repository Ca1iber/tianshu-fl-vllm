# 天数 BI-V150 32g：真实Prefill/混合路径的16/32页验证准备

CPU等待进程160411已实际处于S状态，12份源码冻结并与本地相同。仅等待上游Decode实验159110退出、独立Decode数值与page32验证通过、原始包生成、GPU空闲和服务端口释放后，才运行本候选实验；当前没有启动GPU工作。正式B64S8最后一轮仍优先。

probe_prefill_pages.py使用实际进程hook链、MetadataBuilder.build、TritonAttentionImpl.do_kv_cache_update和forward，初始化顺序与WorkerFL一致：adaptive Decode在hybrid之前。记录实际原生Prefill Router来源，避免32页悄悄回退。依赖源码和真实KV writer源文件另外快照；不加载模型或修改安装源码。

规划9个组件场景：纯Prefill32/33/2048、chunk2048-prefix、ragged、mixed64、mixed-ragged、Decode64和未重排fallback，最长5K。page16/32使用同一逻辑KV但物理页表独立打乱。缓存包含未使用页和非连续物理间隙；待写位置先毒化，真实writer写入后整块缓存与预期逐位比较，额外3个slot=-1必须跳过；输出尾部、额外head与segment临时区尾部保持哨兵。

捕获writer+forward后，在原地址改变query、KV值、写入slot与有效长度，再重放，并对独立GPU FP32注意力及CPU抽样参考比较。两轮输入均保存检查及页16/32差异；7组AB/BA、每组20次图重放设备event计时，保留原始耗时。

本检查点只有AST/CLI和实际CPU等待验证，所有GPU预期仍待执行。若Decode数值未通过则停止此跟进；此脚本失败也不自动启服务。即使组件通过，仍需要真实小4K服务A/B、当前页布局独立完整Math105严格>95，明确提升后才跑原版正式4K；没有新page32精度/吞吐结论。无16K、无权重量化或采样修改、无沐曦源码修改。
