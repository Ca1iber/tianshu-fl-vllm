# 模型原生 FX 接入检查，2026-10-21

完整通过：43张静态64图和43张动态[1,4096]图，分别覆盖全部42个GateUp bank；观察前后图源逐字相同。只加静态compile64和只读post-grad observer，未安装融合算法。两个独立salt的128/16冷请求均成功，缓存命中增量0、无抢占，输出token完全一致。不是吞吐或准确率评测。

真实激活图包含Gate/Up切片、Gate转换FP32、sigmoid和乘法、转换BF16、再与Up相乘。进一步检查本次模型生成代码可见Inductor默认消掉了中间BF16舍入，最终仍在FP32连续相乘，只在输出store转换BF16。因此不能仅凭FX图的中间dtype认定真实执行舍入点。代表IR和生成内核函数体分别见LINEAR_IR_M64_32g.json、GENERATED_ACTIVATION_M64_32g.txt。

模型生成的标量形式为x*sigmoid(x)*y；现有FlagGems融合组件使用fdiv(x,1+exp(-x))*y，FP32算术顺序仍有细节差别，完整模型等价性及新鲜质量仍需验证，不因数学等式而声称逐位等价。

全部86张图、服务响应、环境观测和407源子集快照保留容器 `/workspace/benchmark-results/20261021_nn_silu_fx_observer_GPU_tianshu`。CPU核对synthetic图只证明观察器不改该测试图，真实42层覆盖由GPU启动获得。FX_GATE记录实际图源hash；RAW_SHA256记录原始结果及代表生成代码路径/全文hash。
