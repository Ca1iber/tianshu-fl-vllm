# QKV非对称INT8：纯CPU精度筛选

不改checkpoint、模型结构或正在运行的服务。读取原checkpoint全部42层真实合并QKV（2560x2048），原903个r3源和所有对称W8 pack/scales签名精确核对。比较四种固定方案：原A/W均对称；仅W非对称；仅A非对称；A/W均非对称。对称为amax/127、RNE、[-127,127]；非对称为max(max-min,1e-8)/255，z=clip(RNE(-128-min/scale),-128,127)，q=clip(RNE(x/scale)+z,-128,127)。先RNE再整数z相加，不能用round(x/scale+z)混淆偶数舍入。

统一矩阵补偿公式D=Qa@Qw.T-za*sum(Qw)-zw*sum(Qa)+K*za*zw，逐元素与INT64直接中心化整数GEMM严格一致。K2048下中心化绝对上界2048*255^2=133171200<INT32，所有实际原始点积和修正中间项也检查INT32范围。GPU尚未执行，不宣称torch._int_mm/新epilogue已支持或有收益。额外sum/z计算必须以后计入完整GPU链。

两组独立随机BF16输入M64/K2048，种子202610241000/1001，零第一行，每组取固定8个等距token行；Q/K/V分别等距64/16/16个输出通道，共96列。两组fixture不共享校准或数据拟合。FP64原BF16矩阵乘积作误差参考；各候选INT64修正后转FP32、按A/W scale顺序两次乘法、BF16 RNE再转FP64比较。记录全部42层/每阶段/每方案的误差、全局加权relativeL2、最大绝对误差、权重重建误差及zero-row精确为0。它们是合成独立输入，不是实际模型激活或Math-500正确率；CPU加权误差降低>5%只表示值得另立GPU实验，不能授权模型采用或吞吐。

纯CPU、CUDA_VISIBLE_DEVICES空、Torch仅CPU读safetensors，前后CUDA未初始化。代码/计划/完整输入SHA与原checkpoint前后哈希保留，全部逐层原始数组和完整源快照只存容器。GitHub只代码、报告、关键CSV和CPU门槛。用2个BLAS线程和低优先级，执行前后保存当前GPU评测阶段，不修改/暂停任何GPU进程。
