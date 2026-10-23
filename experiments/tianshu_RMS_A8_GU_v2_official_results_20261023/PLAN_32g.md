# RMS+A8 GateUp v2正式4K独立CPU核验

首次正式GPU前固定源码/门槛。原N256×4、首轮排除，原始逐轮与summary CSV独立重算，全部1024请求成功和长度正确，模型916/原903精确子集、全部控制器与前快照、原benchmark SHA、原checkpoint SHA、实际13参数/63环境均核验。新鲜完整C64/C8各105题必须严格>95%且独立全量CPU质量审计通过，所有逐题原始文件SHA重验。

重新运行全部旧框架/attention/KV/scheduler/sampling/GU/LM/Down/QKV/O及新norm融合审计；实际42层norm和原GU GEMM捕获/二进制、独立A8未伪造、动态范围不改、实际M4096覆盖、真实显存预算、零抢占、就绪后无Triton autotune、结束GPU空闲。仅CPU、无需导入Torch。模型数值/权重不改，原始结果CSV不覆盖。

晋级：正式热三轮mean>5950.75×1.005 total tok/s、极差/mean<1%、TTFT<=3174.58×1.01，加上同期短4K和所有质量/运行门槛。独立历史比较非同时AB，未经本Gate通过不得合格发布。只4K，无新16K或完整赛事资格声明。数组/完整日志只容器，GitHub仅源码报告关键结果和必要压缩控制证据。
