# W8A8 fresh MATH-500 Level 3

2026-10-20，原105题、C8、temperature1/top_p0.95/max_tokens32768，100/105=95.238095%，严格大于官方95%门槛。105请求均有完整原始prediction/review、无API错误；103个stop，2个max_tokens。总输出358306 token，最大32768，中位1440。没有复用旧103/105。

独立核验全部105逐题分数、usage、原始行/文件SHA；Git仅存分数、用量和哈希，完整回答保留天数容器。实际C8 model Graph捕获且回放42个GateUp整数分支，LM head在model Graph外实际执行整数GEMM；全部43组原BF16/INT8/scale哈希与成功screen一致，layer0 GateUp和LM头也与真实权重组件测试一致。原始BF16 checkpoint和embedding未修改，新增非持久权重1326938112 bytes已计入真实FL模型内存profiling。

运行冻结311份来源、9个控制器、57项环境，VLLM_DISABLE_COMPILE_CACHE=1，原13项serve argv。正式4K服务的实际进程环境和同来源配置已独立核对，正在benchmarking；本报告尚不宣称正式4K成绩或adopted。此前正式5235.52 tok/s与其103/105质量记录继续保留，不能将不同profile结果混为同版。

原始目录：/workspace/benchmark-results/20261020_dense_w8a8_math_tianshu。正式目录：/workspace/benchmark-results/20261020_dense_w8a8_official4k_tianshu。该新profile没有完整16K结果。
