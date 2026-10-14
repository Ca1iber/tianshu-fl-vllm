# 天数BI-V150 32g：修复Graph输出强引用后再试Prefill piecewise

实际FL graph.py SHA ba78ed0a依赖weak_ref_tensors释放输出，而_C_ops_registry weak_ref_tensor fallback是lambda t:t。独立ATen非持有别名已经CPU3stride owner释放和4shape变输入Graph各10次逐位检查通过。

仅候选FL GraphWrapper的输出weak_ref_tensors改用别名，保持CUDA Graph私有池对分配的拥有；其他Torch弱引用接口不改。主体算术/输入/采样不变，mem0.70、83点、4096预算、.cv/S8/KV16和原生TopP继续保留。

先单独candidate确认捕获及3轮4K/C64/N64能跑；此前同0.70 control稳态4057.954927仅作临时参考，不冒充新鲜同期比较。若有至少2%且高于旧对照全部稳态，则启动新鲜5+5，再新Math与正式4K。未完成任何新质量或正式成绩。当前最佳4091.29不变。

非持有别名使mem0.70捕获从2/83推进37/83后仍显存不足。新候选保留module的全部83点GEMM warmup和资格/补齐函数，仅将configure后的实际图列表缩为18点：[1,2,4,8,16,24,32,40,48,56,64,128,256,512,1024,2048,3072,4096]。旧tracer configured记录的是configure内83点，最终真实capture必须以worker结束记录18点和capture_policy为准，不能冒充83捕获。未捕获shape由dispatcher向上补齐，工作负载/采样不变。先候选三轮可行性，若有收益须同期新鲜确认与新Math。
