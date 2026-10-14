# 天数32g：正式benchmark采样路径更正

## 发现与原因

当前4096预算正式4K中，API ready后记录到fused_exponential_kernel_f32_unroll8对N7964160（61×130560）的一次0.80秒调优。原“正式benchmark纯贪心、不经过采样/排序”的判断错误，原因是检查了旧vllm/benchmarks/backend_request_func.py；实际CLI serve.py导入vllm.benchmarks.lib.endpoint_request_func。旧判断已标记superseded，无效结论保留以便复盘。

## 当前真实证据

安装serve.py的temperature默认None，只把显式采样参数加入extra_body；当前原官方脚本0813未传temperature/top_p，endpoint payload也不硬编码它们。CLI明确输出no longer sets temperature==0提示。实际模型generation_config.json为temperature1.0、top_p0.95、do_sample true；因此默认随机采样，RNG设备日志与此一致。CPU执行真实endpoint payload构造和common update（截断在网络I/O前），验证温度/top_p均省略；不伪造当前请求的逐次元数据计数。

当前源码快照：endpoint SHA5fd4b66d...、serve SHAb09e85ee...；详见CPU_ROUTING_AUDIT_32g.json的完整SHA及reference_sources。

旧正式最佳3826.24的原始benchmark.log同样四次输出默认温度更改提示、benchmark命令一致、包版本vllm0.24.0+empty一致；本轮没有修改正式脚本、CLI或请求配置。因此正式成绩对照继续有效，等待全四轮后核验。已有小规模4200.39(+3.34%)为temperature0/top_p1贪心对照，不能把采样模式说成与正式相同，也不能直接替代正式成绩。

当前TopP源码在HAS_TRITON且batch>=8时走原生topk_topp_triton，batch<8走PyTorch sort回退。因此沐曦通用sort优化有小batch尾部的适用可能，不能再以“正式不排序”排除；也不能反向声称B64正式主路径使用generic sort。需要实际route/阶段耗时核验后决定GPU实验，不为制造收益切换原生TopP路径。

## 后续实验调整

新8192预算实验尚未GPU运行，已将双方吞吐请求统一到temperature1.0/top_p0.95，贪心rare-shape canary继续0/1。CPU执行真实request函数的payload构造及真实ThreadPool submit实参，验证未来请求值1/.95、canary0/1。当前正式4K和已排队寄存器/缓存/M4096探针的26份冻结源码未改，不重启评测。旧短测基础保留，后续新框架服务对照不得默认沿用其贪心参数。

尚未采profile，不把0.80秒RNG调优当作几十秒GEMM调优或总吞吐瓶颈；本轮完成后仍需按API ready边界逐类统计全部调优。
