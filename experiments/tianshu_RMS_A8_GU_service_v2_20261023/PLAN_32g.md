# r3 + 静态M64 GateUp前IR-native RMS+A8融合

在原r3精确903项模型输入上，仅静态compile_range=[64,64]的42层post_attention fused_add_rms_norm及其紧随的现有NN two-dot GateUp+SiLU替换为功能式自定义op。新op先调用已完成108数值与图链计时独立核验的RMS+A8固定kernel，然后调用原有GEMM+SiLU固定kernel，返回投影结果与残差；不增加模型权重/打包、无全局Tensor工作区，原结构/checkpoint不改。QKV64仍BF16，QKV128/512/4096、O、Down、LM、KV/scheduler全部沿用r3；动态[1,4096]及其余范围保持原图字节。

新组件图链M64带残差6.4892→4.8220us（25.6912%），无残差30.4780%，只是独立编译组件线索，不能代替模型收益或宣称原模型CUBIN一致。当前接入仅使用带残差核；首轮单次Graph未过筛证据保留。

实际42层新op捕获、真实norm和原GU GEMM二进制、图键与后续重放由单独记录审计。为保留原审计中的真实GU GEMM/SiLU覆盖索引，新op把实际原GU kernel调用记入原兼容索引并显式标明原独立A8未执行、wrapper为新源码；这不是旧custom_op函数计数，不伪造旧量化调用。另有独立新记录核对42层新norm和GEMM记录一致。原NN打包/全部其余框架和量化审计保留，原源码不改。

首次GPU前冻结全部模型输入及质量控制器。先启动smoke验证真实图/二进制/42层捕获，再同版新鲜完整C64/C8各105题严格>95%（全部超长/errors计入分母），不可重跑挑分数；通过后同期r3 N64×3排除首轮短4K。仅收益>0.5%、候选波动<1%、TTFT<=对照1.01才正式N256×4。模型收益尚未验证，不运行16K、不自动发布。GPU互斥、2026-10-23T15:58Z硬停。代码/报告/关键结果GitHub与容器，全部大数组/缓存仅容器。

本轮以完整质量控制器在同一次服务启动后的startup audit作为smoke：原全部启动检查和新42层真实捕获/二进制/图范围审计完成后才启动Math，不额外重启或复用旧准确率。

## v2：修复编译schema元信息

首版在Math前编译失败，原始失败保留。仅改新FX节点的meta：新op与两个getitem只赋实际输出val，不复制旧norm的eager_input_vals等旧schema字段。数值kernel、配置、静态M64范围、所有质量/性能门槛不变；不是原样重跑。此版本重新冻结916模型输入，先startup audit，未通过不运行Math。
