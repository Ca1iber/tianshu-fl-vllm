# 当前K8V16/S16地址int32：CPU准备通过，GPU排队

这是针对当前压缩K8/V BF16布局重新评估地址算术。10/13原BF16地址32曾有16K+6.55%、4K-0.64%的历史完整结果；不能直接套用到当前K8V16/S16内核。新父源码为已验证7b94...S16，原件复制为qualified_S16_reader.py用于恢复和哈希核验，不包含权重或张量。

两组候选：kv32仅改K/V六个stride整数注解和两个物理页index的加载后cast；all32改全部28个整数注解、6个整数cast。AST核验只有tl.int64→tl.int32，浮点解码/乘加/softmax、KV值、布局、原始物理64位指针和S16分段保持。两者都要求运行时有符号元素offset（含2048×stride和的保守masked余量）小于2^31。

CPU检查8个真实/边界形状和32000个独立整数地址样本。首次准备计数断言与边界用例修正记录保留在CPU/CPU_v2原件；最终CPU_v3使用仓库内的同SHA父源码。guard.py实际选择器另经9项CPU调用检查：合法范围选择窄地址，宽shape/stride、非3D、其它量化/TD/sink等路径直接用原64位内核，保持参数、grid和返回值；不窄化64位物理pointer，也不修改slot_mapping张量类型。scalar未使用descale只形成offset0。

GPU probe已准备3组同数据×12场景×2相位，含B64/4K与5K、B40/16K、B41跨8192阈值、ragged/stride、负slot、完整writer/cache字节、FP32 GPU/CPU参考、逐位同control、输出/segment guard及RNG。计时为单层KV更新+decode热Graph，9次交替顺序×20 replay；不是服务吞吐，也没有新Math。

run_after_service.sh只在当前小暂存区4K+16K六轮对照全部完成、236源复核、GPU已释放后启动组件；deadline仍为2026-10-20 23:58北京时间。GPU_PREPARATION_32g.json仅CPU/AST证据，当前无GPU数值或速度结论、未采用。GPU成功后仍需依据实际C64收益决定是否做服务筛选，任何采用都需要全新Math严格>95%及原始完整验证。
