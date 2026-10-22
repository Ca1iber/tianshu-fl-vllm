# 2026-10-22 量化探索状态

每天北京时间06:00–次日00:00可用，本日GPU硬停止23:58。代码、报告、关键数据在天数 GitHub `exp/tianshu-quantization-20261022`；原始数组仅容器，没有本地项目副本。

完成295个权重矩阵、5格式、1475行CPU静态扫描。完成6版实际量化组件，180个tile配置、540相位artifact，全部通过独立CPU数值审核；逐元素/成对解包、NN/NT布局、每通道/每组后缩放都测过。现行Down/LM8用BF16，LM64用W8A8；这些候选均未超过对应现用路径。详细比较见 COMPONENT_SUMMARY_32g.md/CSV，各版报告及完整关键数据在各自目录。

原checkpoint未改、架构未转换、未训练。这些是weight-only运行时副本原型，单份压缩字节比不能视为引擎净显存节省。没有组件阳性时未浪费时间跑新模型Math或正式benchmark；尚无新模型正确率和正式吞吐成绩。

IXKN首次导出异常保留；V2直接CSV成功并CPU审核105行/8section，完整候选输出与同一binary保持。V3原生控制也成功，103行/8section，应用前后selected输出一致。确认原生Down/M64实际使用16warps/2stage，旧量化候选主要4warps/1stage；实际指标提示并行度/资源配置差异。新12点只测Down/M64+尾部，包含原生同tile/threads/stages与邻点，正在组件验证，尚无结论。详见../tianshu_w8a16_native_tiles_20261022和两个有效profile报告。

正式发布仍 **5695.95 total /1139.19 output tok/s，TTFT3663.77ms**；官方4K N256四轮排首轮，官方C8 Math103/105、实际C64补充101/105。当前任何量化原型不借用这些质量成绩。
