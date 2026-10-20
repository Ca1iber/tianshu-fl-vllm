# 整数 GEMM 动态 M：减少重复编译的独立组件对照

只更改四处 AST：量化原语删除函数体未使用的 M constexpr 参数及调用实参；GEMM 的 M 改成运行期参数并 do_not_specialize。所有计算体、INT8 点积、缩放、mask 和 launch tile 与已合格 b9b692c 原语一致。原合格源文件不改。

使用原始 layer0 GateUp 和 LM 权重，M1/3/7/8/17/32/37/40/48/56/63/64，额外 M37 行 stride=2055。共26案例×3个改变输入/权重符号及 scale 的阶段；含零行与 RNE 半整数。设备端全矩阵比较两个版本 INT32/BF16 位级一致，并独立 GPU 整数参考；CPU复核全部A和选中原W行重新量化、INT64点积及选中输出。输入、RNG和前后边界均检查。

记录实际 CompiledKernel cache hash 与设备二进制 SHA，要求同一角色/tile 跨 M 真正复用一个 GEMM 二进制、同一 stride 跨 M 复用一个量化二进制；不是只依据 Python 声明推断复用。7轮交替顺序，每轮20次热Graph，包括完整A量化+GEMM。当前 CPU gate 只证明四处 AST 和通用整数界；GPU等价/复用/性能必须实际运行后才算。

等待前一动态批次组件终态后再占GPU。本组件不执行模型，不产生新 Math/正式成绩，不自动推广到运行时。只有无明显热点回退且完整核验通过，才考虑新服务对照。
