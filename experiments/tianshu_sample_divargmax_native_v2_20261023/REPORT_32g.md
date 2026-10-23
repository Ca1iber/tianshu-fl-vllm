# 原生循环融合：GPU一致，完整CPU审计未通过，不接入模型

生成设备代码只在实际FlagGems argmax_inner三个输入load后加入原FP32除法和完整score写回；归约/循环/启发式不改。CPU AST先检查具名参数绑定。8case×2phase的GPU完整商bits、token、图重放、noise/base/RNG检查均通过；连续C64组件含等量输入恢复0.53944→0.46103ms(-14.535%)，strided包含contiguous/写回则0.99122→1.01944ms(+2.847%)。这不是完整采样或模型吞吐。

独立CPU审计失败，原verify_cpu.py与错误日志保留，没有重跑GPU来挑结果。该审计把局部NaN归约误当NumPy max；实际Corex Triton `_argmax_combine`是`value1>value2 or tie`，NaN结果依赖归约树，并不具备该假设。独立部分审计只能证明全部商、非NaN行的CPU token及全行native/candidate token相同，不声称已有NaN CPU oracle。另查到小strided all_zero阶段保存的输入并非全零，故不声称预定这个边界已覆盖；没有证据把原因归结到具体库函数。

因此完整CPU scope仍未过，不接服务、不跑新Math/正式benchmark、不替换r4。两个初次失败分别是自定义首NaN规则不符与原循环启动参数绑定错误，原件均在原目录。完整数组保留容器并归档，Git仅源码、报告与关键审计JSON。
