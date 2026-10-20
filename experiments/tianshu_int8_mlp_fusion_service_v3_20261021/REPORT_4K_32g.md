# MLP融合V3：单趟fallback激活仍未恢复原合格版本性能

不采用V3，不跑新Math或正式N256。有效小4K均值 **5126.403623→5157.656065 total tok/s，+0.609637%**，但候选比原合格版本相同客户端N64参考 **5410.385242** 低 **4.671186%**。control/candidate两轮极差占均值 **2.216597%/1.916155%**，超过1%筛选限值。不能将组件13.67%提速当成模型收益。

两臂均改为在新的MLP opaque边界内对非融合M调用真实 `FlagGems silu_and_mul_out`；candidate仅M8/M64走整数GateUp+SiLU融合，control不融合。模块原来选择的 `SiluAndMul.forward_native` 仍记录为元数据，但在该custom-op路径中被显式覆盖。本次fresh Math未执行，旧正确率不借用。

## 小4K逐轮

C64、N64、4096输入/1024输出；每臂3轮，首轮剔除。所有384条计时响应的usage/正文/时间均保留并独立核验。六轮新增autotune均0，0 preemption。

| arm | 轮 | Total tok/s | TTFT ms | 计入 |
|---|---:|---:|---:|---|
| control | 1 | 4830.893634 | 14368.950667 | 否 |
| control | 2 | 5069.587773 | 12037.010154 | 是 |
| control | 3 | 5183.219473 | 12033.313323 | 是 |
| candidate | 1 | 4834.105379 | 14399.679596 | 否 |
| candidate | 2 | 5108.241733 | 11982.900828 | 是 |
| candidate | 3 | 5207.070397 | 12053.618696 | 是 |

平均TTFT为12035.161738→12018.259762ms（-0.140438%）。本次观测没有运行时autotune记录，不能以缓存漏shape解释全部波动。

## 实际执行证据

568运行时源码、63实际环境、13项原serve argv；两臂冻结源码相同，除模式和输出根目录外环境相同。43份原BF16/packed/scale权重签名与合格版本完全一致，原Down投影及LM小批BF16/M64INT8策略保持。

M8/M64真实42层融合capture与模型replay核验通过，实际编译二进制均与组件测试一致，0 spill。所有42层的prefill GateUp ready路径及实际FlagGems fallback源、SHA和调用记录均核验。独立audit重读与完成时保存结果完全相同；所有源快照重新核哈希。

两臂KV blocks同为40698，capture内存记录也相同：allocated before27662473728、after27663683072 bytes，reserved before28162654208、after27871150080 bytes，进程peak27950029824 bytes。该事实不足以解释吞吐波动原因，未采集硬件occupancy/HBM计数。

## 决定

新边界的两个实际服务实现V2/V3均没有超过原合格参考；停止这一接入路线的继续扩大测试。结论限于本次两个接入实现，不证明所有GateUp/激活融合都无收益。正式最好仍 **5482.79 total tok/s，Math101/105**，没有更新分数、tag或发布入口。

后续回到原合格版本做预算4096/8192框架对照，不包含本次MLP改动。完整16K未跑。

原始容器目录：`/workspace/benchmark-results/20261021_int8_mlp_fusion_v3_service_{control,candidate}_tianshu`；runner为`20261021_int8_mlp_fusion_v3_pair_runner_tianshu`。关键结果在SCREEN_GATE/SCREEN_ROUNDS，完整日志和响应只留容器。
