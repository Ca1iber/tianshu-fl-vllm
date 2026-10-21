# 8192预算完整4K5608.40，无稳定提升，不替换推荐版

原N256/C64、4096输入/1024输出、四轮排首轮；1024请求全部成功。全新标准Math105题103/105=98.095238%，所有生成stop、无请求/评分错误。实际原13参数、445运行来源/64环境/16主控制、43权重bank及CPU/GPU Graph路径经独立复核，与新Math相同；完整16K未跑。

|轮次|total tok/s|output tok/s|Mean TTFT ms|汇总|
|---|---:|---:|---:|---|
|1|5481.96|1096.39|5696.30|排除|
|2|5582.83|1116.57|4427.89|计入|
|3|5616.21|1123.24|4321.79|计入|
|4|5626.15|1125.23|4469.03|计入|
|后三轮原脚本均值|5608.40|1121.68|4406.24|独立CSV/日志/Decimal复算通过|

相对已发布4096预算prefill-INT8版本5586.28/TTFT3780.63，吞吐+0.395970%，TTFT+16.5478%；后三轮吞吐极差/均值0.772413%，高于这次历史均值差。历史比较不是同次配对AB。8192此前同次N64配对小测+1.376484%，不能把这个幅度外推到N256。公开4K性能门槛与>95%正确率均通过；不替换推荐版的依据是尚无稳定吞吐收益、TTFT回退及请求期冷点，并非正确率不够。保留本完整实测记录。验证器的1%/TTFT1%是保守默认选择偏好，非官方规则；连独立的正收益高于本轮极差条件也没有满足。

启动后实际四条autotune：fused_exponential_kernel_f32_unroll8的7,703,040个float32元素，1.05秒（59*vocab130560）；_index_linearized_jit_function三个缓存bucket1888/4416/4224，实际benchmark长度1865/4398/4213，分别0.45/0.24/0.30秒。没有新增GEMM/NT调优。这些冷点值得单独处理，但日志不能证明全部波动/TTFT差由它们造成。原筛选小测六轮零autotune，不代表N256新调度形态也无冷点。

实际KV容量40,178blocks，Math同为40,178；早先8192N64筛选40,219，原因未建立。零抢占、FutureKV虚拟预留全部释放。packed payload1,326,938,112bytes真实计入DeviceMemoryProfiler，99native预热点/40INT8预热形状，所有43bank与筛选/Math签名一致。未知未热INT8尺寸仍走合格基线；当前额外RMS、staticcompile、KV bound准备代码均不属于此结果。

OFFICIAL_RAW和SUMMARY保持原CSV字节/CRLF；COMPARISON_COMPACT只移除4MB运行profile，原完整comparison SHA另外记录且原件留容器。独立额外CPU验证器自有SHA，不改16主控制或其完成状态。原始目录/workspace/benchmark-results/20261021_prefill_budget8192_official4k_tianshu；Math目录同前缀_math_tianshu。推荐tag仍tianshu-original-command-4k-prefill-int8-20261021。
