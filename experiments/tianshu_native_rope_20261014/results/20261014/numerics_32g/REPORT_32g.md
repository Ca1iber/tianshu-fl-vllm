# 天数 BI-V150 32g：真实native RoPE融合40场景逐位通过

baseline模型实际编译图0b5dea52a6显示42处RotaryEmbedding.forward_static原生路径：cache.index_select→BF16四个mul→sub/add→cat。新opaque CUDA算子合并Q16头/K2头、full Neox rotary128、cache gather和计算；支持实际QKV切片stride2560及普通连续Q/K。保留int64 positions与cache地址算术、各乘法先BF16舍入再sub/add，enable_fp_fusion=false；functional新输出，不修改Q/K/cache。opaque fake输出使Inductor将该调用保留为leaf，避免旧RMS裸Triton的FakeTensor问题。

40场景：M1/7/64/576/2048 ×packed QKV/contiguous ×random/zero/large/small；位置包含131071/0/4095。全部逐位匹配当前native和独立CPU显式BF16中间舍入参考，Graph与eager逐位相同，输入/缓存保持，输出无错误别名；key=None正确回退原方法。真实原生源base SHA c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5。动态guard不支持的形状/风格/dtype回退原static method。

实现源SHA 04d617914ab7a6541f1530d7156415b5f96937912483b41381a248400b7d7132，实际记录确认packed stride2560/int64位置/FMA关闭。原始包2360421bytes/10分片，SHA256 1a5d424beba03eaa1e90f16a2acaa8e7776c6f329a4e7e6f50fa88f6f1a09425；全部分片/整包、40唯一场景、真实记录与实现/hook/probe3份源逐字本地核验。原件service-results/20261014_tianshu/native_rope_numerics/20261014_native_rope_numerics_tianshu。

结论：数值验证通过，允许进入实际模型绑定，再4K小规模同期对照；本报告没有模型吞吐或质量新分数。基础保持正式最好TopP16/Decode32/B75G1/budget2048，不叠加已确认只有0.23%收益的IR RMS；正式最好3506.84/Math103不变。若服务ROI明确再新Math105>95和正式4K。性能剖析：TODO图内kernel覆盖；不跑16K，安装源/权重/原benchmark不改，沐曦只读。
