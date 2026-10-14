# 天数32g：当前KV16/warp1+.cv的有界int32地址

旧地址32测试是KV32/warp2/S4布局且主负载慢约6%，不能代替当前KV16/warp1+.cv/S8。本次固定现有producer配置/采样无关数值，比较原64、仅producer32+原reducer64、producer和reducer均32。从真实cv源8deb生成，28步长注解和6整数offset casts改32，物理指针/浮点AST不变，地址类型归一后的完整AST相同；每次候选launch检查所有实际tensor的元素offset与2048 stride masked余量<2**31，超界拒绝。

CPU执行真实probe adapter和已有LaunchRouter/_ReductionRouter配合假JIT，确认三组实际选择的producer/reducer模块、tile/warp/stage不变、只有对应地址源替换，超界fixture被拒绝；不是GPU证据。初次CPU检查捕获跨方法局部变量引用和本地原件路径两处错误，已修正重跑通过，未先运行GPU。

GPU计划17场景含4K/5K、strided K/V、0/1 query填充和tile边界，保留FP32/CPU、原始及改变QKV后的Graph重放、尾部与9轮交替全Attention event，实际双阶段模块与界限记录。候选reducer不修改warp/stage，保持默认4/3。尚无新服务、Math或正式分数；当前4091.29最佳保持。
