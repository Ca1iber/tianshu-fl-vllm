# 软件8位KV解码的binary32中间表示（仅CPU准备）

当前软件8位KV在真实4K/N64服务筛选中约提升3.15%，正在独立做Math105质量验证。本目录仅准备后续独立组件试验，不修改该次验证的任何源文件。

假设：当前uint32位运算→uint16→BF16 bitcast的lowering可能保留较高寄存器开销。替换为构造低16位为零的binary32位模式，再转BF16。256个码值（含正负零）逐位等价，编码/固定scale不变；CPU独立标量参考、65536编码不变、producer/reducer AST不变与可逆源变换均通过。性能和寄存器改善尚未测量，不能把假设当结果。

计划在当前版本质量与正式4K完成后，独立比较BF16/当前byte decode/binary32 decode，实际writer、metadata/forward、变输入同Graph、量化后参考及输出bitwise诊断，包含B1..64与大批次2D、prefill staging与容量fallback。仍须计入全部writer/staging开销。先看组件，再决定服务和新的质量验证；本目录尚未执行GPU或部署。
