# dotsaddz Down64真实42层对照

逻辑量化权重、输入fixture、误差阈值和计时流程与冻结5d80154的AWQ Down42组件相同；物理格式改为刚通过30假设完整16384输出GPU验证的dotsaddz唯一匹配：W int32 [6144,256]沿N自然顺序，S BF16 [48,2048]，Z BF16 [48,2048]=-8*S，q uint4为clamp(round(W/S),-8,7)+8。数学q*S+Z，实际 aten._weight_int4pack_mm_dotsaddz(...,size_n=2048)。仍是group128 RTN，非激活感知校准。

同一r3 903来源、同一实际BF16模块和hook栈为对照，M64真实42层不同bank轮换。所有42层CPU完整逻辑/物理roundtrip；0/20/41层三phase全64x2048原始和BF16反量化权重FP64参考，共2359296格。阈值首GPU前固定：逐格<=.012*abs(ref)+.006*RMS(ref)，relL2<=.006，全有限。全部42层三phase eager/graph poison完整字节一致，完整输入/权重/scale/zero字节/版本/guard/RNG不变；checkpoint不修改。先冻源码、控制器和CPU输入，再GPU。

图与eager各7轮，每轮10次42层，交替顺序，第1轮排除。该版与BF16直接配对，旧AWQ数字仅作不同运行的参考，不当作此次paired结果。不跑Math500或正式benchmark；有足够绝对收益才集成。单次900秒上限、23:58北京时间前结束，失败保留，不放宽阈值或事后换参考。所有大数组仅容器，代码/报告/关键结果推天数GitHub。
