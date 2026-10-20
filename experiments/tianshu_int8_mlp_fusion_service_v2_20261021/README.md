# 整数GateUp+SiLU接入真实LlamaMLP

基于当前5482.79合格配置，GUP8/64整数、LM小批次BF16/64整数、原BF16预填充精度与down投影保持。真实模型是LlamaForCausalLM；42个LlamaMLP在原生加载结束后CPU关联原bank，不重新打包或删除原BF16权重。

两臂都把GateUp+activation放进不透明custom op。control调用原GUP和实际activation对象；candidate仅M8/64使用已验证整数融合，其他M调用原路径。M8补到32，M64不补；复用原4-byte INT dummy的BF16 view作为不写入诊断占位，无额外GPU dummy。融合输出只返回有效行，再调用原down_proj。

编译边界改变可能改变原activation的编译/舍入路径，不能由FlagGems组件逐位一致直接断言整个旧模型不变。记录实际选中的activation来源/qualname、42层真实Graph捕获重放、实际kernel binary、分配/对齐、43银行签名。必须新Math，且最终仍与5482.79正式版比较。

先原serve参数的小4K/1K/C64/N64两臂各3轮排首轮。除配对收益外，候选也须超过此前同客户端合格profile小测参考，避免只相对变慢的wrapper对照得出正结论。不默认跑16K。raw完整输出/日志仅容器，GitHub只代码、报告、关键数据。

V2 fixes only the early capture record selection: candidate checks fused_launches, control checks the original launches. V1 was stopped during startup before any benchmark round; its frozen files and raw state are retained. GPU fusion source and activation policy are unchanged.
