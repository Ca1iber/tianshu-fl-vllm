# K8V16同页标量索引：组件无收益，不采用

当3D decode的TILE_SIZE与16-token物理页相等时，候选明确用一次页表标量读取再广播，替代向量表达式；其他路径、64位页号/偏移及全部浮点运算不变。CPU131072项地址等价、源码精确逆变换和函数AST范围检查通过。

首次入口在数值/计时之前被哈希拒绝：原始cache reader误传给要求已生成K8 reader的变换。旧probe/CPU清单及错误日志留`20261020_k8v16_scalar_page_tianshu`；修正为两组均先生成并验证K8 reader，再变换候选，在独立v2目录运行，kernel变换本身没有修改。

有效v2：10场景×两配置×两个同Graph变数据相位，**40项完整GPU FP32及416项CPU向量检查通过**；所有候选输出还要求与控制逐位相同，完整writer存储、输出/scratch哨兵和RNG通过。9×20 AB/BA Graph计时含一次KV更新和单层decode。

B64/4K控制0.589566ms、候选0.589490ms（+0.0129%吞吐）；B64/5K控制0.729024ms、候选0.729140ms（-0.0160%）。没有实际收益，不接服务或Math。可能编译器原本已消除重复读取，但未据此声称底层汇编完全相同。

原始结果/源快照保留容器`/workspace/benchmark-results/20261020_k8v16_scalar_page_v2_tianshu`。GitHub保存DECODE_GATE_32g.json及离线核验入口；正式4K5101.45、16K3213.11不变。
