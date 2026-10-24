# K4 uint16 word读取：正确但明显负收益

保持前版10496B页面、FP32 scale、BF16 V和精确BF16位构造，仅用一次uint16读取四个nibble，再展开为原128维布局。实际GPU helper覆盖全部65536个word，两种stride和两阶段同Graph，1048576个输出逐bit一致。独立CPU FP64复核完整2768896个attention输出，并按输入重建6454798336字节缓存，全部哈希一致。全部GPU输出、参考、计时和实际binary保留。

| 场景 | 当前K8完整 ms | K4 word完整 ms |
|---|---:|---:|
| C64 4096 | 0.55751 | 3.16600 |
| C64 5120 | 0.68921 | 3.94707 |
| C64 ragged/strided | 0.54635 | 2.87637 |
| C33 17408 | 1.36719 | 6.49257 |

完整8场景时间见WORD_KEY_RUNS_32g.csv，均为9次交替、每次20次Graph回放的中位数。预声明的C64三个主要场景均未达到至少1%收益，所有场景都更慢。失败门槛不改，不接模型、不跑Math、不替换r5。编译器实际资源与转换数量另存WORD_CODEGEN_32g.tsv，组件正确不代表服务收益。原生Spec计数已核对，但模型allocator没有接入K4。

持久归档路径与SHA见WORD_RAW_ARCHIVE_32g.sha256，归档已经tar逐文件比较并核对result.json原字节。代码、门槛、关键结果与报告提交GitHub；大体积raw仅保留容器。
