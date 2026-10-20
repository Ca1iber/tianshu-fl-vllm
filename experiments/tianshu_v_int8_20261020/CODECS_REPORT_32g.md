# BI-V150 32G：线性 INT8 V 的 GPU 编解码与 writer 检查

结论：固定 V 缩放 1/32、1/16、1/8 三档均通过真实 GPU 编解码、完整 cache storage 与改变输入的 writer Graph 检查。此阶段没有 attention forward、引擎 spec 接入、性能、服务或 Math 结果，adoption_allowed=false。当前正式最好仍为 K8V16/default4096 的 5101.45 total tok/s，不以本检查声称吞吐提升。

## 代码与机器

受测源码来自提交 7a1f21637512c1d4abc1b75ffe85903a741ee631；执行时 HEAD 4d3b0820e73527b3bbc70b578c274a3709b158b3 仅补充同源码报告。逐文件摘要见 GPU_CODECS_GATE_32g.json，源快照及原始字节/tensor fixture 留容器。

机器 Iluvatar BI-V150 32G，Torch 2.10.0，IX-ML 4.4.0 / Driver 4.5.0；进程 2563。GPU 检查记录起止为 2026-10-20T01:28:58.174621+00:00 至 01:29:01.908822+00:00；此前还有 Python/Torch/插件初始化，不能把该时间作为冷启动性能。

## 通过范围

每档均检查全部 65536 种 BF16 位模式的实际 V 编码，等于 CPU scalar 规范；实际 K 编码也逐一等于原有软件8编码。每档 K/V 各 256 个解码码点通过，K 包含 signed-zero 位模式。每档 783 个 FP32 nearest/ties-to-even 与饱和边界通过，编码输出尾部 guard 保持。

实际写入使用 8192 字节 UINT8 页和另一档带 128 字节起始偏移、8448 字节物理 stride 的布局；两者都隔页取 view，输入行也带 stride。每档两种布局、每种三个 Graph 阶段，总共 18 次重放，修改 K/V 与 slot 后，全份原始 storage 等于独立构造参考，负 slot、未引用页、间隙和尾部未被改写。此 Graph 仅覆盖 writer，没有 Q/block-table/attention Graph。

三档均 complete=true；GPU_codecs_match_CPU_contract、GPU_K_bits_preserved、GPU_writer_Graph_changed_inputs_passed 均为 true。新 helper 的真实 Triton 编译/执行已发生，但完整 attention kernel 尚未执行。

## 对照与下一步

对照使用独立 CPU 数值规范及原有 K 编码；这里没有把不同近似精度的 V 与原 BF16 数据断言逐位相等。没有计时对照，不能比较旧最佳或官方 baseline 的性能。没有 model V 饱和率或 Math，固定缩放尚未选定。

下一步以同进程 K8V16 为控制，验证真实 attention、dense prefill/fallback、实际页分配和改变 Q/K/V、table、lengths 的 Graph；正确性通过后测 B1/8/64 的 4K/5K 组件耗时。服务提升达到门槛后才进行新 Math 和正式 4K，不常规跑 16K。

原始数据：/workspace/benchmark-results/20261020_v_int8_codecs_tianshu。GitHub 只保存本报告与 CPU/GPU 关键 JSON，不上传完整原始 fixture 或历史缓存。

## 接入恢复说明

09:27 公网 DNS 地址的严格主机密钥校验连接成功，随后原域名的新连接也成功；两条控制连接均保留。不能仅凭时间顺序认定是 DNS 问题或原样提交公钥修复了故障；当前已能正常进入同一容器。
