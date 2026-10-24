# 拟合 QKV/O64 联合模型：CPU 就绪，尚无模型质量或性能结论

本版本在原 r4 上组合已各自完成 GPU 组件资格核验的拟合 W4 QKV/O，仅 FULL uniform64 decode 生效。组件 Graph42层累计 O1.23656→0.84544ms、QKV1.46686→1.15529ms；两个组件独立FP64核验3538944/4423680输出元素，不等于联合模型正确率或吞吐。

CPU准备已完成：全部84组原始权重/packed/scale/zero参数绑定对应组件，84份选定payload文件SHA通过；checkpoint全文件SHA仍14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c；已有392项范围检查及其9份源码哈希重新绑定。原944份模型来源逐项保留，合计995份来源加CPU_GATE自身为996，64项profile/65项实际环境，原13项serve参数不变。CPU_GATE的model_ready只表示可开始GPU模型验证，model_quality_qualified=false保持不变。

原BF16/W8全部保留，额外派生缓冲210567168字节（200.8125MiB），控制/候选同装载，不作净显存减少结论。运行时将在实际DeviceMemoryProfiler内分配，并在实际profile内覆盖84组native调用；审计逐层核对真实buffer字节、原权重版本、实际Graph覆盖、全部模型storage以及继承的GU/Down/LM/prefill/KV路径。

新的run_math.sh先全新API跑C64完整105题以覆盖实际Graph64，再全新API跑C8完整105题。使用官方temperature1/top_p.95/max_tokens32768，全部题和到上限输出计入分母，任一批次≤95%停止推进。C64仅未来KV准入设control，确保实际满批覆盖；其余配置一致，不借用O minmax的102/105。完整105验证函数只复用代码，输入必须来自新目录的预测与判题。尚未启动本联合模型，本文不报告其质量或速度。

O minmax新输入N128对照已稳定完成但仅+0.704%，见../tianshu_W4_O64_fresh128_20261024/REPORT_32g.md，不升级正式测试。联合版本若两批质量通过，再按同样跨轮新seed协议筛选；明确提升后才完整N256，不默认跑16K。当前正式最好仍r4 6050.94 total tok/s。

后续存储修复：首次实际落盘在API启动前因2.4GB SDK副本耗尽磁盘，未执行模型；详见../tianshu_W4_QKVO64_fitted_results_20261024/REPORT_INITIAL_PREP_FAILURE_32g.md。v2改为复用3个已校验的独立SDK捕获副本，在真实工作盘完成新预检；来源数更新为996+CPU_GATE=997，量化参数与初版逐字节相同。新GPU输出使用fitted_v2_math64/math8目录，不覆盖失败记录。
