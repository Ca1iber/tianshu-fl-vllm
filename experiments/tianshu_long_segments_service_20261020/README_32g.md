# 长上下文分段服务接入：准备验证

保持已验证K8V16/4096/原始serve argv，两组同样83预热、43项基配置；只新增三项模式/记录目录并替换PYTHONPATH。原始205源不改。依据B25..64完整扫描，仅B33..56允许maxS16，GPU实际上下文≤8192保持四个活跃段。原Reader的Python dispatch和writer不动，Worker初始化完成后、模型预热/Graph捕获前替换producer/reducer的两个JIT对象和明确的分段选择函数。

复用已验证的实际FL GraphWrapper仪表，两组都记录每个Graph的真实capture/replay、attention捕获配置及现有CPU长度；不读取GPU Tensor，不增加随机操作。核验须包含新核源SHA、选定批次的S16捕获及同Graph重放，不以旧upstream wrapper替代。

CPU621项实际旧policy选择对照、对象绑定/原Python dispatch身份和语法检查通过。GPU服务、模型质量与benchmark结果尚未产生，未采用。先做4K/1K C64 N64三轮排首轮的服务对照，核实无回退；长上下文收益需要同工作负载单独验证。结束时等待GPU真正空闲并保留首次/最终硬件快照。
