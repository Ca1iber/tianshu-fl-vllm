# 新模型：Down64 W4保持，LM扩展M1..64 W4

在原Down64+LM64 W4候选基础上，仅将LM group128 RTN W4覆盖扩为1<=M<=64。M1/2引用已验证direct W4小批次组件，避免M1补到32多约30%耗时；M3..32补32，M33..63补64，M32/64本身直接，使用已冻结完整64M×3phase验证的kernel。大于64或不满足原dtype/contiguous条件委托原CUDA backend；所有GU/QKV/O/K8V16/scheduler/sampling/Graph75点设置、原13serve参数和63项完整环境保留，唯一profile差异PYTHONPATH。Down64同父版42层W4，其他Down范围同r3。模型架构和原checkpoint不改，不训练/微调，不借用旧Math。

复用LM真实WD/S4/ZD载荷142049280字节，原BF16和W8副本保留。真实loader DeviceMemoryProfiler内装载、全部文件/数组/原weight/W8签名验证，真实native determine_available_memory.profile_run内显式探测M1/2/31/32/63/64及pad临时buffer和native workspace峰值；无全局Tensor workspace或自设KV字节，不声称净显存节省。warm/profile不改CPU/GPU RNG，全部model buffers纳入实际accounting。

使用原opaque自定义算子的public register_kernel，仅替换LM1..64 CUDA路由；保留原schema/真实bank identity，不额外改EmbeddingMethod.apply。runtime记录每个真实M和bucket、source/kernel SHA、阶段、capture标识、真实Python调用数，不把Python数冒充Graph replay。

首次GPU前必须同步冻结C8/C64/full/Down审计：M8旧BF16要求显式替换为真实W4 M8证据，M64旧W8要求替换为真实W4 M64，原BF16/W8 counter不修改不造数据；保留全部GU、KV、attention、prefill、调度、原生memory profiler及源/环境检查。C64复用已明确的Down W4审计适配，避免已知旧LM64 INT8断言错误。

先fresh完整105题Level3 C8和C64，官方>95%即至少100正确，所有caps/errors都计入105。C8这次实际改变LM8精度，不能借用原LM64候选C8成绩。两者通过并独立复核后才能同期r3/新候选N64 C64短4K三轮排首轮，gain>.5%、spread<1%、TTFT<=control*1.01；再视结果执行原始N256正式4K四轮排首轮，vsr3 5950.75需>.5%、spread<1%、TTFT<=3174.58*1.01。GPU测model吞吐期间CPU完全空闲，仅轻量状态查看。不跑16K，不原样重跑挑成绩。

硬停止2026-10-23 15:58UTC(23:58BJT)。代码/报告/关键JSON/CSV推天数GitHub，全部原始仅容器，没有本地项目副本。当前只是新模型候选，不是已采用版本或质量/速度结论。

CPU首稿仅显式warm六个代表M，尚未运行GPU。最终v2将loader和native profile中的预热都扩为完整M1..64（所有pad REAL constexpr），把64种Triton编译和原生分派成本移到启动；bucket仍为direct1/2、pad32/64，不改数值或运行白名单。审计必须验证完整64个M预热记录，避免动态批次运行时新增pad形状编译。首稿CPU记录保留，只有本v2允许准备GPU质量评测。
