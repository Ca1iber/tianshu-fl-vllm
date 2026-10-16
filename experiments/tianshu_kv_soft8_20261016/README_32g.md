# 软件8-bit浮点KV：分离读取编码转换成本

此前INT8不论scale外提或warp/segments均比BF16慢。此处不同假设：保持半字节数的KV读取，改用uint8存储自定义有限E4M3（scale1/16），用整数位操作直接构造BF16 exponent/mantissa，避开有符号整数转浮点路径。不是平台原生FP8支持，也不是原生KV dtype；手工cache、尚无writer/真实allocator/模型/校准/Math或服务成绩。

编码为符号1、指数4/bias7、小数3，storage scale1/16；所有非零指数有限，最大30。指数零编码flush signed zero，包括8-bit subnormal。BF16输入以RNE丢4个尾数位、进位保留、溢出clip到30；只对finite输入定义此组件量化。固定scale未对真实层校准，fixture无clip不能推出模型无需校准。独立frexp/round参考检查全部65280 finite BF16编码和256个解码含符号零；GPU另外直接调用实际_cast helper测试全部256编码。

生成器在当前8deb .cv源只加uint8/量化mode0的cast分支和独立256码测试函数，producer/reducer AST不改，不使用INT8 scale分段改写。B64 4K/5K页内交错，两阶段改变Q/K/V，清空输出重放原Graph；对自身反量化BF16的FP32和CPU参考，BF16近似误差另记，尾部保护；3项tile/warp配置与原BF16，7交错组各20 replay。所有量化写入成本排除，不能直接声称服务提速。明确组件收益后仍须真实所有KV流程接入/通用校准/新Math105>95和原正式4K。23:58 GPU硬停，无16K，不修改已冻结目录、最佳启动器、权重、安装源或沐曦。

实跑完成：tile16/w1仅4K-0.896%/5K-1.383%组件耗时，regs244对85；tile32/w2有125 spill。65源/56事件/全部256实际GPU解码均已本地核验，写入未计，不接模型/新Math，详见results/20261016/micro_32g。
