# P量化：只隔离每行倒数后的乘法

从完整INT8 PV负结果独立派生，不修改该失败原件或r5。三组：当前K8/V BF16 S4/T16/W1、原严格整数PV S4/T32/W1、新每行倒数整数PV S4/T32/W1。页表、8448-byte V8实际布局、writer、K编码、QK、online softmax、reducer、segments/warp/stage及其余舍入一致；只替换P量化。scale仍为div_rn(peak,127)，新式先r=div_rn(1,scale)，再nearbyint(weighted*r)，原式nearbyint(div_rn(weighted,scale))。FP32双舍入可能改变个别量化码，明确为新精度合同；不是逐码等价承诺。先核对独立标量FP32/PythonINT64、随机/相邻ties/极小/零五阶段，全输出精确匹配各自公式，量化误差预定≤.5001scale。

完整B64/4K/5K/ragged-strided，保留两阶段同Graph改变Q/K/V/scale/table/lengths/slots，整份raw/input/output/scratch/RNG保护，实际writer样本独立CPU重算，fullGPUFP32及所选CPUFP64 reference。原.002+.01abs(reference)门槛不改；若失败，断言前保存全部actual/reference和最多8个失败head的Q/K/V，以便独立复核。计时仍9正反交替×20replay，含writer，实际IR/ASM仅warmup采集，计时不做独立CPU准备。

只有新固定配置相对当前K8/V16在4K/5K均≥5%组件收益且全部数值/保护通过才推进；仅相对原慢整数版加速不算达标。后续仍需更广batch/prefill/engine、新C8/C64Math>95%、fresh4K和原完整benchmark。单独记录相对原整数版的变化，不据静态除法指令归因动态瓶颈。失败和负结果保留；GPU截止2026-10-24 23:58BJT。代码/报告/关键结果GitHub，完整数组只在容器。
