# QKV M64 NN整数布局：现行471源直接对照

组件QKV M64 NN .0271881ms vs原BF16 .0325848ms（+19.85%），完整34场景/102相位和9400320个整数输出由原checkpoint独立CPU重算。
只启用M64，M8 NN负收益不启用；O负收益、Down仅+5.8%低于预设10%组件门槛，不混入本候选。
模型candidate保留原BF16权重，在原生DeviceMemoryProfiler内打包42层QKV physical[K,N]，新增220631040bytes并披露真实KV容量。
新整数M64必须在真实CUDA Graph捕获时启用，实际42层capture/kernel binary与Graph replay都要核验；不能捕获后开关冒充生效。
未知M/非连续/有bias走原方法，不扩展其他精度白名单；opaque候选可能改变编译上下文，不预称所有fallback逐位相同，最终需新Math质量。

直接reference运行现行471源推荐版原入口，提前开始；它不加载新hook、不包装原apply。
candidate之后运行新的数值源图；其原471文件必须逐字节为子集，原13参数、62配置归一化后只有PYTHONPATH不同。
两臂客户端均采用原N64三轮排首轮、4096输入/1024输出/C64，temperature1/top_p.95，保留完整请求usage与seed canary。
控制器客户端源码应与现行原入口逐字节相同。不同数值manifest是预声明的代码改动，不宣称两臂manifest相同。
稳定热均增益>.5%才继续新官方C8 Math105与实际C64新路径Math105（各严格>95%），然后原N256四轮；不常规测16K。
原结果/失败保持；不为随机质量重采样。23:58北京时间GPU硬停，代码/报告/关键结果存GitHub和容器。
