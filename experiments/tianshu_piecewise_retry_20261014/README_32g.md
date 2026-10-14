# 天数BI-V150 32g：Prefill piecewise CUDA Graph服务对照

control为当前.cv/4096预算/83点/S8/KV16/nativePrefill/nativeTopP配置、FULL_DECODE_ONLY。candidate仅改FULL_AND_PIECEWISE；额外进程级bucket资格代理让该模式保留原有模型/并行/预算保护及83点预热，padding/configure/warmup函数不变。attention仍动态执行，不声称完整Prefill图安全。logits gather依赖同一bucket资格，因此继续启用。默认maxseq256、temperature1/top_p.95和模型长度131072不变。

CPU执行真实native+8192扩展guard的180个正/负资格用例，4096个ceiling边界；真实嵌套CLI拦截通过。尚待GPU启动与实际piecewise模式/83warm记录、canary及3+3服务，预排除首轮。>=2%+区间不重叠再独立确认、新Math和正式4K。可能出现CoreX piecewise支持问题，保留失败并确保清理GPU；不改安装源码，不改现有最佳启动器。

初始hook与已有扩展bucket Finder相互委托产生递归，原组在模型启动前失败。重试独立目录保留原17份冻结源；新的Finder在委托期间暂时摘除自身，finally恢复顺序。CPU执行实际Finder AST，对两个目标模块的成功/异常四分支验证仅一次委托且恢复完整。
