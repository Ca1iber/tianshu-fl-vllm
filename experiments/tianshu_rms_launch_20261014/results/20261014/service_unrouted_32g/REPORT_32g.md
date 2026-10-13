# 天数 BI-V150 32g：RMSNorm服务未命中，停止无效对照

控制组保留TopP w16/stage1、Decode L0/KV32、B75G1/budget2048，3轮4K/1K/C64/N64，预排首轮。控制轮1315.768554、3531.613407、3535.117936 total tok/s，计分均值3533.365671；192个已完成请求均成功。候选首轮1311.658008/TTFT18955.62198ms/249.821217s，64个请求成功；进入第二轮后主动停止。没有完整候选warm均值，没有可用ROI，也不新增模型分数。

候选API PYTHONPATH正确、TIANSHU_RMS_LAUNCH_PROBE=1，TopP实际57组w16/s1路由、Decode64组tile32路由；但RMS优化记录始终0。实际EngineCore日志明确IR op priority为rms_norm=['native']、fused_add_rms_norm=['native']。vLLM0.24模型层forward_native调用vllm.ir.ops.rms_norm或fused_add_rms_norm.maybe_inplace，绕过本次FlagGems RMS module.kernel替换。没有证据表明选择了vendor.iluvatar RMS实现：FL策略实际仍prefer flagos，不能把vendor源码的存在当作真实命中。

发现目标未运行后，仅SIGTERM本实验父进程72806，候选API/engine清理，GPU空闲。父及候选result.phase='stopped'，候选输出保留部分记录，归档明确intentionally_incomplete=true/valid_roi=false；未伪造complete结果、未将首轮比较当优化收益，未运行新的Math或正式4K。

原先80项边界证明的是直接调用gems_rms_forward及候选原Triton函数体的数值行为，并不证明模型走该路径。下一步必须优化实际IR native调用链并另做数值/alias契约验证。尤其实际IR fused在FP32累积后先转weight dtype再乘weight，旧Gems fused kernel直接FP32乘weight，两者BF16舍入路径不同，不能复用旧80项为IR语义完全相同的证明。

完整停止原始包3596637bytes/14分片，SHA256 ddc41e66e4d885a5e0fa5d74b458170a40e5250b50c767a60ddeebaa9334c29f，全部分片/整包、6份冻结源码、4个已完成64请求轮、两组安装源和真实日志均本地核验。原件service-results/20261014_tianshu/rms_launch_unrouted/20261014_rms_launch_service_tianshu。源码保存这次实际未命中尝试，后续重试使用独立路径和代码，不追改本次结果。

结论：停止此绑定路线，保留数值微测结果，但不采用候选；当前正式最好仍3506.84/TTFT4494.12/Math103，tag tianshu-best-topp-w16-20261014。性能剖析：TODO，IXsys已知Graph调用覆盖正在另测。安装源、权重及官方benchmark保持原件，沐曦只读。
