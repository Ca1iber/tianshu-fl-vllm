# K-only 分块物理布局：独立组件准备

探索KV cache内部布局：page16、BF16不变，只把K每页[token,head,dim]存储改为[head,dim/x,token,x]，x=2/4/8/16；V保持原NHD。原有HND实验仅交换token/head，未改变dim内部布局，本试验不是重复HND。

读写两处地址共同变换，K/V页容量、block table、采样、attention乘加/归一化、CV/L0、当前KV16-w1/S8和原native prefill 256/128配置不变。所有生成文件独立，不修改安装包或最优源码；限定page16、2个KV头、head128、原NHD strides、BF16且无TD/quant。CPU验证4×4096地址双射、逆映射和K/V物理半页互不覆盖，尚不是GPU正确性或收益。

计划8场景：B64/4K、B64/5K含stride、B32非等长、B1页边界、33token prefill、2048 prefill、4096历史chunk2048、ragged prefill。每个场景5种布局，2阶段原地改变Q/K/V及适用的长度/slots，同一CUDA Graph重放；完整物理KV存储、输出padding、segment scratch尾部、RNG不变，候选输出必须与控制逐位相同，并对FP32/CPU参考抽查。7轮交错顺序、每组20次重放，共280原始event，计入writer与attention。

这只是使用相同kernel配置的组件试验，不声称已经接入真实模型、backend dispatch、scheduler或完整prefill/decode混批流程。GPU组件通过且有稳定收益后，才值得实现全框架cache布局协议。当前TopP服务3+3对照进行中，必须等待其进程结束与GPU空闲再串行启动。不可据此更新4093.81正式成绩。
