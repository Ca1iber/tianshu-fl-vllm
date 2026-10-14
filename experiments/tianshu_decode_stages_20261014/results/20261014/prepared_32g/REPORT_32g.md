# 天数32g：准备当前KV16/warp1的流水阶段诊断

旧扫描只覆盖KV32/w2/stage2、KV16/w2/stage2，未覆盖当前KV16/w1配B64S8的stage2/3。因此按现有最优tile/warp/grid/分段不变，单独比较编译num_stages1→2/3。主循环仍使用原生for range，是否能有效流水化必须靠真实编译与计时判断，不假定编译选项一定产生预取。

CPU实际Router AST与生产LaunchRouter执行573组，覆盖B1..64、稠密及两种0/1查询长度、stages1/2/3，逻辑query网格、reducer tile和num_stages参数传播通过。GPU计划17场景×2候选、独立FP32/CPU及改变query后的Graph检查，保存实际num_stages/shared与所有事件。无数值或吞吐结果。

仅在当前GEMM服务3+3完整归档并终止且GPU空闲后启动。如果当前服务达到2%稳态门槛，优先留给其独立确认，流水探针停在held状态。
