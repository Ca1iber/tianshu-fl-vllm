# 待继续：真实服务接入约束

当前全42bank holdout GPU与独立CPU均通过。两M的NN/ROWS1固定候选含Down仍M512+6.16%、M4096+18.07%，值得真实4K。不要停在组件阳性，不借旧Math；正式5695.95保持。每天北京时间06:00–24:00，GPU硬停止23:58，不需再问预约。

可新增独立服务目录/分支配置，复用正式471源、62环境、13argv，new hooks先runpy现用m_range_prefill_service_v2/hooks_candidate/sitecustomize.py，再安装新WorkerFL finder。不要编辑旧服务/源码/用户脏文件或沐曦代码。

现用post_grad_custom_post_pass为RangePass，parent后在动态range[1,4096]把原linear/SiLU转换为torch.ops.tianshu_m_range_prefill.fused_or_original(x,weight,packed,scales,bank_id)。可追加pass，仅将该target换新同签名customop；fake返回同BF16 shape。新op只在candidate+contiguous M512/4096使用quantize+torch._int_mm(q,packed.T,out=integer)+epilogue ROWS1，其他M调用旧range customop，保留decode和所有fallback。packed逻辑[N,K],stride(1,N)，packed.T已NN contiguous，不重打包/不留NT副本。

native整数workspace是每调用临时torch.empty([M,12288],int32)，epilogue Projected未用时传out(2D BF16保留SME资格)。不要全局side-channel缓冲；Torch stream allocator负责生命周期。private临时分配的峰值必须被native profile_run覆盖；不能把active开关放到capture_model之后才开启，因为Worker.determine_available_memory先调用profile_run再配KV。新candidate在profile_run就应真正执行M4096所有42bank；记录profile阶段native实际调用和peak，不能借用仅warm的计数。若需预热persistent cublas/module开销，可在42nd NN weight处理且原DeviceMemoryProfiler仍活动时warm，无新增权重payload；动态profile仍需新路径。

CoreX控制cubin有debug路径问题：设备函数V1/V2 AST相同，但完整cubin不同。3点已复现M128取V1、M512/4096取V2，其余旧range warm共23M仍需严格匹配。新全23M来源复现任务将开始，见tianshu_prefill_control_binary_all_20261022；勿替换旧/tmp cache。可用匹配private cache对原range kernel的同一JIT对象做一次启动预热，确保后续其device_caches命中匹配binary；必要时在load_model期间42nd bank处理内完成。所有23M正收益原路径仍会在旧capture wrapper后warm，不能只处理新native两个M导致其他M校验失败。full-binary gate保留，GPU来源诊断不当模型质量或性能。

优先复用已有服务candidate_profile/audit_profile和数值/route trace，冻结新增源、真实env、argv及正确继承旧Math门禁；新的Math必须fresh C8全105与实际C64全105>95%=至少100。短4K N64三轮排首轮，官方N256四轮仅在稳定>0.5%后；不要常测16K。每轮报告推GitHub，raw只容器，不在本机建工作副本。要核对所有生成/实际命中数值路径、GPU idle、source freeze和RNG，无法证明实际命中就不能声称收益。
