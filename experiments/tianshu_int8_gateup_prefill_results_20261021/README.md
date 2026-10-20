# Prefill GateUp INT8 组件离线复核

仅 CPU，不启动服务或 GPU。继承原真 INT8 验证器的完整 A 量化、选中实际 GPU INT64/BF16、源快照和七轮交替计时检查，场景限定 M128/256/512/1024/2048/4096 的六个 GateUp NN 分块。额外逐行回读原 safetensors，验证保存原 W 行确实来自 checkpoint。所有近似误差相对当前 prefill BF16 linear_contiguous；这些组件误差不会替代最终模型 Math。
