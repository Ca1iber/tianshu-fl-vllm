# 天数 BI-V150 量化探索结论，2026-10-22

本轮没有产生可替换现用路径的新量化优化。9版权重组件共300配置、900次变化输入相位全部完成独立数值审核；另完成A8倒数修正的GPU域枚举和6种shape性能对照。最接近当前Down/BF16的是W4A8 NT，对称group128完整路径0.075430ms，当前BF16同进程0.058882ms，仍多约28.1%耗时。没有把负优化接入模型，没有新Math或正式benchmark成绩。

正式可交付版本保持：官方4K N256四轮排首轮 **5695.95 total / 1139.19 output tok/s，TTFT 3663.77ms**，官方C8 Math103/105，实际C64补充101/105。这些质量/吞吐结果只属于已有发布版，不属于本次新原型。已有GateUp/LM64 W8A8与K8/V16 cache继续有效，本轮负结果不否定已采用的量化优化。

## 做了什么

先对全部295个矩阵做5格式、1475行CPU扫描：W8每通道、W4对称/仿射group64/128。W4载荷包括scale/zero约为BF16的26.6%–28.9%，但如果保留原BF16 fallback就是新增副本，不能声称引擎净显存节省。静态权重误差不等于任务正确率。

随后使用实际checkpoint第0层Down/LMhead权重、合成BF16 activation，逐步测试：逐元素/成对解包、NN/NT布局、W8A16每通道后缩放、W4A16每组后缩放、按原生GEMM配置调整warps/stages、W4A8整数分组dot，以及NN路径的量化/GEMM拆分计时。W4采用4bit存储后展开BF16或INT8计算，未宣称原生INT4指令。

|关键实验|候选耗时|同进程现用控制|结论|
|---|---:|---:|---|
|W8A16 Down/M64，原生配置引导的最快tile|0.090219ms|BF16 0.058685ms|原型改善明显，仍多53.7%耗时|
|W4A8 NT Down/M64，对称group128|0.075430ms|BF16 0.058882ms|仍多28.1%耗时|
|W4A8 NN Down/M64，对称group128|0.147719ms|BF16 0.059135ms|NN本版更慢|
|上述NN仅矩阵本体|0.143239ms|BF16 0.059135ms|省去独立量化约4.6us仍不足|
|A8 M64/K6144，half_div修正倒数|5.505us|原量化器4.581us|数值一致但变慢|

前6版完整对照见 [COMPONENT_SUMMARY_32g.md](COMPONENT_SUMMARY_32g.md)。全部9版的实际目标最优点及现用控制见 [FINAL_COMPONENT_COMPARISON_32g.csv](FINAL_COMPONENT_COMPARISON_32g.csv)。逐tile数据、源码、冻结计划和独立审核均在相应实验目录；表中不同实验来自各自控制，不跨进程宣称细小收益。

IXKN实际采集也已完成：首次导出进程异常及无法导入的binary保留；改为直接CSV后取得有效候选/原生控制，审核105/103行计数器。原生Down/M64为BM64/BN64/BK128、16warps/2stage，旧候选主要4warps/1stage。依据该证据测试12组新tile后W8A16从0.226373ms降到0.090219ms，仍未超过BF16。计数器只支持资源/并行度差异的线索，不能单凭占用率断言唯一原因。详见 [profile报告](../tianshu_quant_ixkn_v2_20261022/REPORT_32g.md)。

A8检查发现直接乘倒数会在半整数边界改变量化码，CPU构造域出现7424处差异。两种边界修正通过GPU每种81113088元素的完整q/scale和独立CPU审核，但6种性能shape都慢。范围为有限BF16域；枚举元素含重复零/填充，不等于81M个唯一组合。详见 [A8报告](../tianshu_a8_reciprocal_20261022/REPORT_32g.md)。

## 证据与限制

最终CPU汇总重新核对全部源冻结hash、GitHub关键结果与容器raw结果hash、每版CSV、独立审核和checkpoint：300配置、900相位，798048个selected FP64输出；两版W4A8合计3133344个实际INT32 partial与独立INT64 dot相同。原checkpoint仍为5033557096字节，SHA256 `14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c`，未转换架构、训练、微调或改写模型文件。

完整decode/FP32参考、BF16 cast、selected CPU FP64/INT64、A8码和scale、复用同一Graph变化输入、尾部/strided、输入/guards/RNG均已验证。组件正确不等于新模型质量；这些新路径未跑模型，不能借用现版Math。失败的signed-zero参考、离线序列化和首次profiler导出均保留原记录，未覆盖。

结论局限于本轮实现、shape与tile，不能推断所有BI-V150低比特实现都无收益。更低的压缩载荷不保证更快；在本版W4A8 NN中，独立计时已经排除“只融合掉A8量化就能跑赢”的假设。当前没有值得继续投入完整模型评测的组件阳性，因此本轮量化探索收尾，现版代码和质量证据保持。

最终交叉核对可在容器执行：

```bash
cd /workspace/vllm-plugin-FL
CUDA_VISIBLE_DEVICES='' python3 experiments/tianshu_quantization_survey_20261022/summarize_final_cpu.py
```

此命令不使用GPU，会读取原checkpoint与已保留的raw目录并更新本报告旁汇总JSON/CSV。重跑GPU实验应使用新输出目录、新冻结版本并更新预约截止时间，不能覆盖本次历史目录。每天北京时间06:00–次日00:00可用，当前硬停止23:58。代码、报告、关键JSON/CSV在天数GitHub研究分支；原始数组、profile和完整日志只留容器，没有本地项目副本。
