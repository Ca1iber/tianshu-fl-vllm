# V成对uint16加载正确，但没有胜过现用K8/V16

在E5M2/FP16_PV已有最优S4/T32/W1仍慢17.7%的基础上，只改变非TD V加载：按uint16读取两个byte，直接还原两个FP16bit pattern并interleave；V赋值直接使用FP16结果，避免再经过原helper回退BF16。量化/写入规则、物理8192-byte候选页、PFP16、K/QK/softmax/reducer都不变。新源码的精确逆变换与AST范围证明通过，适用范围严格为head128/padded128、V最后stride1、even-byte stride/对齐、无TD。

GPU两个tile分别检查全部65536个word→131072个rawhalf bit pattern，含符号零/Inf/NaNpayload；另外检查masked最后一行、尾部guard、输入与RNG。全部65536 BF16 writer/K编码及256码解码分类也重跑通过。三场景×四路径×两个同Graph变数据阶段，共3145728个完整GPU FP32参考输出元素和36864个所选CPU FP64标量通过；全raw cache/input/output/实际segments scratch/负slot/RNG保护正确。所有路径实际配置、冻结来源/快照与TTIR/TTGIR/LLVM/ASM已独立核验。

|B64热Graph，ms|现用K8/V16 T16|旧V8 scalar T32|新pair T16|新pair T32|
|---|---:|---:|---:|---:|
|4K上限|0.589564|0.692676|0.753973|1.245497|
|5K上限|0.729117|0.856158|0.934135|1.550969|
|ragged/strided|0.579979|0.664481|0.725458|1.131314|

全部S4/warp1/stage1；定长第二阶段长度减7，实际计时4089/5113。9次交替顺序×20Graph replay包含实际KV更新与一层decode，参考在计时外。新pair T16比现用控制多耗时27.9%/28.1%，pair T32多111.3%/112.7%；旧scalar T32仍是byte候选最优。没有只挑相对旧较差配置的局部改善来晋级。

实际n_regs/n_spills：控制142/0，scalarT32 253/0，pairT16 174/0，pairT32 256/12；shared分别8192/8192/8192/12288B。T32新增spill是实际资源代价，T16无spill仍更慢，因此这些记录不能单独解释全部硬件慢因。静态编译代码不是动态硬件计数。

没有固定配置在4K/5K都胜过控制5%，不进入引擎/Math/正式benchmark。没有改r5，当前正式best6123.31 total tok/s。原始参考记录/源码/IR/ASM/event都留容器并归档逐文件比较，路径和SHA见RAW_ARCHIVE_32g.txt；GitHub只保存代码、报告与关键结果。未序列化全部tensor，不声称离线重算完整attention。

下一步先拆开存储与计算精度：考虑同样12KiB物理页的原生FP16 V存储，与当前BF16 V比较，不再同时加入byte压缩/解包。该方向尚未测试，也没有服务或Math结论。

对照范围补充：本实验使用7b94历史合格标量K读取器作组件控制；正式r5使用f1df成对K读取，不能将本组件ms等同r5实际attention成本。详见[当前KV基线说明](../tianshu_framework_status_20261024/CURRENT_KV_BASELINE_NOTE_32g.md)。原始数据、门槛及未采用状态不改。
