# K4/V16 v1完整组件：正确，读路径负收益

在当前r5 f1df paired K8/V16上，只改K编码/读取与逐token/head缩放。真实writer+reader+reducer同一Graph，9次两组交替计时、每次20回放；单独reader/writer只定位开销，不替代完整组件结果。源码、实际二进制、grid、寄存器、spill及全部逐次时间保留。

| 场景 | K8完整 ms | K4完整 ms | K4耗时倍数 |
|---|---:|---:|---:|
| C8 / 4096 | 0.08357 | 0.41394 | 4.9533 |
| C33 / 4096 | 0.50257 | 1.66515 | 3.3133 |
| C64 / 4096 | 0.55770 | 2.82259 | 5.0612 |
| C64 / 5120 | 0.68946 | 3.52568 | 5.1136 |
| C8 ragged / strided | 0.09472 | 0.36301 | 3.8324 |
| C64 ragged / strided | 0.54711 | 2.55604 | 4.6719 |
| C33 / 17408 | 1.36736 | 5.88983 | 4.3075 |
| C64 capture1 ->5120 | 0.68961 | 3.52015 | 5.1045 |

写入均约0.012 ms，额外耗时集中在reader。所有8场景、2阶段、2组都通过完整GPU FP32参考及独立CPU FP64全输出复算（2768896个输出元素）；按seed重建全部原始BF16 K/V/Q、页表与长度后哈希一致，并独立重建每组每阶段整个raw缓存的字节与哈希。全部raw页、padding、guard、输入、负slot、输出、scratch与RNG保护检查通过。相同Graph原地改变Q/K/V/页表/长度/slot，另有捕获长度1后重放5120的覆盖。精确计数及每项误差见ATTENTION_GPU_GATE_32g.json。C33是独立未补齐组件，不把它当作r5 actual33->Graph40的端到端成绩。

当前K4实际组件页10496B，相对12288B减少14.5833%；原生FullAttentionSpec可表达该预算，但未接模型allocator，不能声称已释放模型显存。r5源码、checkpoint及服务配置不变，未跑此候选Math或完整benchmark。

对应实际C64二进制的TTGIR/LLIR/ASM保存在raw归档的codegen_evidence。现行K8为8192B共享内存、176寄存器、0spill；K4为12288B共享内存、113寄存器、0spill，主循环多了K的local_alloc/local_load；LLIR还出现int16->BF16数值转换。两边QK/PV dot均存在，metadata.use_sme是位域，不把非零当成是否启用某一种矩阵指令的证明。这些是回退线索，尚不能单独证明因果；下一独立变体只替换小整数的等值BF16位构造，检验能否避免转换和额外共享内存。

原始result保留pending独立FP64字段；随后CPU复核的完成依据是ATTENTION_GPU_GATE_32g.json及其raw_result_sha256，未改写原GPU记录。此版本全部场景负收益，不满足组件收益门槛，不接模型、不采纳。原始fixture由固定NumPy PCG64 seed生成、全输入哈希冻结；保存全部GPU输出及GPU FP32参考，以支持独立重建和FP64复核。持久盘归档及SHA见ATTENTION_RAW_ARCHIVE_32g.sha256。
