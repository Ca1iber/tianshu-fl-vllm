# 首次接口导入盘点被CPU-only检查拒绝

2026-10-22 20:47，CUDA_VISIBLE_DEVICES=''下import vllm._custom_ops/vllm.ir.ops会加载fl平台插件和FlagGems。原程序在任何schema查询或Gate生成之前发现torch.cuda.is_initialized()==True，assert失败，原始代码与/sourceSHA和容器完整失败日志保留。没有生成CPU_INVENTORY，不能宣称CPU-only通过或原生INT8融合已支持。

日志同时显示未能导入标准vllm._C，但这不单独证明服务插件不会注册其他provider。未调用本研究的GPU数值kernel、未改任何模型/框架/915质量源码；单独查询进程已退出。后续新V2仅以标准库读取文件/AST，不导入torch/vllm/FlagGems，明确原生schema和运行能力未查询。失败日志 /workspace/benchmark-results/20261022_rms_quant_fusion_inventory_CPU_tianshu.log，不上传完整日志。
