# 同一K4数值/存储，替换整数到BF16转换

v1的8场景全部慢3.3–5.1倍，GPU全输出和独立CPU FP64及全缓存重建通过。生成IR显示K路径多了一次4KiB共享内存local_alloc/local_load，另有32处int16到BF16 sitofp。不能仅凭静态IR归因，因此只变一项：对精确范围[-8,7]构造对应BF16位，而不作数值类型转换。K量化、页面、scale、QK/PV/softmax、writer、grid和配置全部沿用v1。

先CPU穷举16个整数精确位模式，再GPU运行实际生成helper的全部256个packed byte（包括writer保留的code0=-8），连续/跨步、两次原地变数据同Graph，逐bit输出及guard/RNG检查。通过后沿用v1的全部8场景、两组9次交替20回放，GPU FP32全参考，随后独立全输出CPU FP64和全缓存字节重建。保留v1全部失败结果；新variant不修改v1源码或收益门槛。

核对新的TTGIR中K local_alloc和LLIR sitofp是否消失，以及实际binary/共享内存/寄存器/时间。该优化只有数值与完整组件收益通过才考虑接模型，随后重新全量Math>95和4K；不从组件推定端到端收益，不改模型权重或架构。当前r5保留不变。
