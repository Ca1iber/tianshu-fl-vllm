# BI-V150 32G：保持 INT8 V 编码的 magic-float 解码探索

前一版线性 INT8 V 在 8 个 decode 场景全部正确，但 batch64 耗时约增加32%，寄存器142→183；实际汇编新增8→16→32位有符号扩展链。原始反例和测量源码保持不变，参见 ../tianshu_v_int8_20261020/DECODE_REPORT_32g.md。

本轮仅改 V 读取 helper，保留相同二进制补码、三档缩放、writer、K codec、attention 数学和启动器。令 b=code xor128，把 0x4b000000|b bitcast 成 FP32，即精确的2^23+b；计算 FMA(value,1/D,-(2^23+128)/D)，输出严格等于 signedINT8(code)/D。全部256码点需 CPU 穷举，且必须真实 GPU 码点和 attention 验证，不以代数解释代替硬件检查。

只探索是否避免该后端生成的 signed-conversion 链，未预先认定寄存器、带宽、占用率或性能会改善。当前没有本轮 GPU 或吞吐结论；服务 allocator、prefill、服务和 Math 未实现/运行，adoption_allowed=false。正式最佳仍为K8V16/default4096，5101.45 total tok/s。

check_cpu.py、probe_codecs_gpu.py、component_launch.py、probe_decode.py 复用上一版实际检查代码，改动只在新的 codec.py。CPU_GATE_CONTAINER_32g.json 保存容器实际运行的 CPU 门槛，GPU gate 明确记录 CPU runner 为天数容器；相关冻结输入包括上一版 codec.py。原始文件留 /workspace/benchmark-results，代码、报告与关键结果推天数 GitHub；运行硬截止仍为北京时间2026-10-20 23:58。

## CPU 与首轮 GPU 检查通过

容器 CPU 检查通过全部 BF16 位模式、舍入边界及三档各256个 magic 解码码点等价检查。GPU gate 的有效输出为 /workspace/benchmark-results/20261020_v_int8_magic_codecs_tianshu/micro_v2：三档 V 编码/解码、K 位模式、FP32 边界、完整跨页 storage 和18次 writer Graph 重放均通过。CPU 证据来自天数容器而非新 GitHub CI。首次预检查失败及修正范围见 PRECHECK_NOTE_32g.md；未放松数值断言，原失败日志保留。

GPU 编解码通过仍不代表 attention kernel 已验证，更没有吞吐/Math结论。下一轮运行同进程 K8V16 对照的8场景 decode gate，原始事件和源码冻结写入独立输出目录。
