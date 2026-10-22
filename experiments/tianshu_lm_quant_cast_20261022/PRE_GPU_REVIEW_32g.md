# 首次GPU前复核

r3实际LM64 binding显示actually_inside_capture=false、graph_key=null，因此后续资格筛选改为eager组件链>3%，不能只凭graph加速授权模型实验。图计时继续保留诊断。补充全部计时结束后的输入/权重/scales/RNG与边界哨兵检查，明确七轮排首轮、后六轮中位数。首版CPU源/Gate/执行清单保留在Git61bf083和容器preGPU_review目录；本组件尚未运行GPU。新runner_v2为独立空目录，原r3及正在运行的928质量源码均未变。

继续源码复核确认r3 dense binding的kernel_sha256是kernel.py源码SHA（b9b692cf…），不是CUBIN SHA。首次GPU之前将对照检查明确为源码SHA+实际配置+93寄存器/零spill，并单独记录本次实际CUBIN哈希；不把源码哈希冒称设备二进制证据。原r3没有提供该字段的CUBIN参考，不声称跨编译缓存CUBIN逐字节相同。加入SIGALRM/TERM/INT的显式失败记录；仍按相同绝对截止/十五分钟上限，raw失败不得覆盖。runner_v3为最终纯CPU准备目录。

首次GPU前补足声明的72个CPU抽检列：在64个等距列与边界列去重后，加入32767/32768边界，固定为72列；八行、三阶段共1728个CPU INT64/BF16→FP32单元。完整GPU比较仍是每项每阶段8,355,840元素，不受抽样影响。最终runner_v4与全部源/Gate重新冻结。
