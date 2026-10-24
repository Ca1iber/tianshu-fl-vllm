# V-only E5M2：编码CPU穷举通过，GPU与模型尚未验证

保持当前K软件E4M3与Q/K计算，只压缩V。V使用FP16高字节对应的E5M2格点，最近偶数舍入，饱和到±57344；NaN映射保留符号的零。实际writer不产生8个Inf/NaN保留码。不是修改checkpoint或依赖原生FP8计算。

标准库CPU证明独立遍历65536种BF16位模式：65280个有限值、256个非有限值均与不经过FP16转换的frexp实数舍入参考一致。256个解码码点逐个检查：248个有限码可精确表示为FP16/BF16，含正负零全部往返一致；另8码为IEEE非有限分类。正常范围内未饱和值的最坏相对量化误差为1/9≈11.11%，该精度代价需要新的Math验证，不能由CPU正确性证明替代。

两个reader只替换V cast调用并增加helper，K helper、Q/K score、softmax、reducer和外部接口AST不变。BF16_PV解码回BF16；FP16_PV解码为FP16，因此已有P.to(V.dtype)使P×V的P类型变为FP16，此精度差异单独记录。Writer只替换V编码，保留K编码与slot/负slot逻辑。生成代码均通过语法编译和逆变换源码一致性检查。

CPU检查没有导入Torch/Triton，也没有访问GPU。现有联合QKV/O服务保持原配置，E5M2没有接入allocator、prefill、模型或服务。后续先测GPU全部编码码点与writer+decode组件；只有B64/4K和5K都有至少5%减时才考虑模型接入。

原始生成源、CPU结果与5个冻结来源快照已归档容器；tar逐文件比较通过，路径、SHA与大小见CPU_RAW_ARCHIVE_32g.txt。GitHub仅保存代码、证明、报告和索引。此CPU工作在联合量化候选服务启动阶段完成，不与服务吞吐计时并行。
