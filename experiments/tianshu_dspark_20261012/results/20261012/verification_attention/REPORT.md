# DSpark 验证阶段的原生注意力数值诊断

原始 eager DSpark 与相同 target 参数非投机版本在第3个输出token分歧。检查实际 multi-token native 2D 配置 M16/TILE32/w4/s3，场景为 Q8/K21、Q8/K23、Q8/K4096、双请求Q8/K21和K23、混合Q1和Q8/K21和K23。固定相同BF16输入、随机分页及实际交错KV缓存，输出与FP32参考比较，并独立抽查CPU FP32与尾部保护。

原实现1/5通过，短序列4项未通过（最大绝对误差0.5363–0.9334，relative RMS约20.5%–22.2%）；无sink softmax分母显式从0初始化后5/5通过，最大绝对误差不超过0.00693。两版本各28/28独立CPU参考检查通过，5/5输出尾部保持。此处确认的是修正前后数值差异，尚未证明底层编译指令原因，也尚不能证明服务输出分歧已完全解决。

原生源码SHA bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0；生成修正源码SHA 4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e。服务诊断使用独立开关，在进程内只替换2D kernel，保留原始3D decode和所有默认tile/launch配置。安装源码及原模型保持不变。

完整原始、生成源码、source bundle及环境记录均已本地逐块和整包SHA核验。L1包1830599 bytes/SHA4b516a99f3095881add16cb226591422cce7fd6d564e6a494193d82ad5015f84；L0包1865014 bytes/SHAb1683597b6c035572b2e51b9cdb013909aea0c7964d85759eab6ade7a730cbfe。不是Math-500或性能测试，不更新2712.55 tok/s正式最佳。
